---
url: /dev/tutorials/intermediate/4_PINN2DPDE.md
---
# Training a PINN on 2D PDE {#Training-a-PINN-on-2D-PDE}

In this tutorial we will go over using a PINN to solve 2D PDEs. We will be using the system from [NeuralPDE Tutorials](https://docs.sciml.ai/NeuralPDE/stable/tutorials/gpu/). However, we will be using our custom loss function and use nested AD capabilities of Lux.jl.

This is a demonstration of Lux.jl. For serious use cases of PINNs, please refer to the package: [NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl).

## Package Imports {#Package-Imports}

```julia
using Lux,
    Optimisers,
    Random,
    Printf,
    Statistics,
    MLUtils,
    OnlineStats,
    CairoMakie,
    Reactant,
    Enzyme

const xdev = reactant_device(; force=true)
const cdev = cpu_device()
```

## Problem Definition {#Problem-Definition}

Since Lux supports efficient nested AD upto 2nd order, we will rewrite the problem with first order derivatives, so that we can compute the gradients of the loss using 2nd order AD.

## Define the Neural Networks {#Define-the-Neural-Networks}

All the networks take 3 input variables and output a scalar value. Here, we will define a wrapper over the 3 networks, so that we can train them using [`Training.TrainState`](/api/Lux/utilities#Lux.Training.TrainState).

```julia
struct PINN{M} <: AbstractLuxWrapperLayer{:model}
    model::M
end

function PINN(; hidden_dims::Int=32)
    return PINN(
        Chain(
            Dense(3 => hidden_dims, tanh),
            Dense(hidden_dims => hidden_dims, tanh),
            Dense(hidden_dims => hidden_dims, tanh),
            Dense(hidden_dims => 1),
        ),
    )
end
```

## Define the Loss Functions {#Define-the-Loss-Functions}

We will define a custom loss function to compute the loss using 2nd order AD. For that, first we'll need to define the derivatives of our model:

```julia
function ∂u_∂t(model::StatefulLuxLayer, xyt::AbstractArray)
    return Enzyme.gradient(Enzyme.Reverse, sum ∘ model, xyt)[1][3, :]
end

function ∂u_∂x(model::StatefulLuxLayer, xyt::AbstractArray)
    return Enzyme.gradient(Enzyme.Reverse, sum ∘ model, xyt)[1][1, :]
end

function ∂u_∂y(model::StatefulLuxLayer, xyt::AbstractArray)
    return Enzyme.gradient(Enzyme.Reverse, sum ∘ model, xyt)[1][2, :]
end

function ∂²u_∂x²(model::StatefulLuxLayer, xyt::AbstractArray)
    return Enzyme.gradient(Enzyme.Reverse, sum ∘ ∂u_∂x, Enzyme.Const(model), xyt)[2][1, :]
end

function ∂²u_∂y²(model::StatefulLuxLayer, xyt::AbstractArray)
    return Enzyme.gradient(Enzyme.Reverse, sum ∘ ∂u_∂y, Enzyme.Const(model), xyt)[2][2, :]
end
```

We will use the following loss function

```julia
function physics_informed_loss_function(model::StatefulLuxLayer, xyt::AbstractArray)
    return mean(abs2, ∂u_∂t(model, xyt) .- ∂²u_∂x²(model, xyt) .- ∂²u_∂y²(model, xyt))
end
```

Additionally, we need to compute the loss with respect to the boundary conditions.

```julia
function mse_loss_function(
    model::StatefulLuxLayer, target::AbstractArray, xyt::AbstractArray
)
    return MSELoss()(model(xyt), target)
end

function loss_function(model, ps, st, (xyt, target_data, xyt_bc, target_bc))
    smodel = StatefulLuxLayer(model, ps, st)
    physics_loss = physics_informed_loss_function(smodel, xyt)
    data_loss = mse_loss_function(smodel, target_data, xyt)
    bc_loss = mse_loss_function(smodel, target_bc, xyt_bc)
    loss = physics_loss + data_loss + bc_loss
    return loss, smodel.st, (; physics_loss, data_loss, bc_loss)
end
```

## Generate the Data {#Generate-the-Data}

We will generate some random data to train the model on. We will take data on a square spatial and temporal domain $x \in \[0, 2]$, $y \in \[0, 2]$, and $t \in \[0, 2]$. Typically, you want to be smarter about the sampling process, but for the sake of simplicity, we will skip that.

```julia
analytical_solution(x, y, t) = @. exp(x + y) * cos(x + y + 4t)
analytical_solution(xyt) = analytical_solution(xyt[1, :], xyt[2, :], xyt[3, :])
```

```julia
grid_len = 16

grid = range(0.0f0, 2.0f0; length=grid_len)
xyt = stack([[elem...] for elem in vec(collect(Iterators.product(grid, grid, grid)))])

target_data = reshape(analytical_solution(xyt), 1, :)

bc_len = 512

x = collect(range(0.0f0, 2.0f0; length=bc_len))
y = collect(range(0.0f0, 2.0f0; length=bc_len))
t = collect(range(0.0f0, 2.0f0; length=bc_len))

xyt_bc = hcat(
    stack((x, y, zeros(Float32, bc_len)); dims=1),
    stack((zeros(Float32, bc_len), y, t); dims=1),
    stack((ones(Float32, bc_len) .* 2, y, t); dims=1),
    stack((x, zeros(Float32, bc_len), t); dims=1),
    stack((x, ones(Float32, bc_len) .* 2, t); dims=1),
)
target_bc = reshape(analytical_solution(xyt_bc), 1, :)

min_target_bc, max_target_bc = extrema(target_bc)
min_data, max_data = extrema(target_data)
min_pde_val, max_pde_val = min(min_data, min_target_bc), max(max_data, max_target_bc)

xyt = (xyt .- minimum(xyt)) ./ (maximum(xyt) .- minimum(xyt))
xyt_bc = (xyt_bc .- minimum(xyt_bc)) ./ (maximum(xyt_bc) .- minimum(xyt_bc))
target_bc = (target_bc .- min_pde_val) ./ (max_pde_val - min_pde_val)
target_data = (target_data .- min_pde_val) ./ (max_pde_val - min_pde_val)
```

## Training {#Training}

```julia
function train_model(
    xyt,
    target_data,
    xyt_bc,
    target_bc;
    seed::Int=0,
    maxiters::Int=50000,
    hidden_dims::Int=128,
)
    rng = Random.default_rng()
    Random.seed!(rng, seed)

    pinn = PINN(; hidden_dims)
    ps, st = Lux.setup(rng, pinn) |> xdev

    bc_dataloader =
        DataLoader((xyt_bc, target_bc); batchsize=128, shuffle=true, partial=false) |> xdev
    pde_dataloader =
        DataLoader((xyt, target_data); batchsize=128, shuffle=true, partial=false) |> xdev

    train_state = Training.TrainState(pinn, ps, st, Adam(0.005f0))

    lr = i -> i < 5000 ? 0.005f0 : (i < 10000 ? 0.0005f0 : 0.00005f0)

    total_loss_tracker, physics_loss_tracker, data_loss_tracker, bc_loss_tracker = ntuple(
        _ -> OnlineStats.CircBuff(Float32, 32; rev=true), 4
    )

    iter = 1
    for ((xyt_batch, target_data_batch), (xyt_bc_batch, target_bc_batch)) in
        zip(Iterators.cycle(pde_dataloader), Iterators.cycle(bc_dataloader))
        Optimisers.adjust!(train_state, lr(iter))

        _, loss, stats, train_state = Training.single_train_step!(
            AutoEnzyme(),
            loss_function,
            (xyt_batch, target_data_batch, xyt_bc_batch, target_bc_batch),
            train_state;
            return_gradients=Val(false),
        )

        fit!(total_loss_tracker, Float32(loss))
        fit!(physics_loss_tracker, Float32(stats.physics_loss))
        fit!(data_loss_tracker, Float32(stats.data_loss))
        fit!(bc_loss_tracker, Float32(stats.bc_loss))

        mean_loss = mean(OnlineStats.value(total_loss_tracker))
        mean_physics_loss = mean(OnlineStats.value(physics_loss_tracker))
        mean_data_loss = mean(OnlineStats.value(data_loss_tracker))
        mean_bc_loss = mean(OnlineStats.value(bc_loss_tracker))

        isnan(loss) && throw(ArgumentError("NaN Loss Detected"))

        if iter % 1000 == 1 || iter == maxiters
            @printf(
                "Iteration: [%6d/%6d] \t Loss: %.9f (%.9f) \t Physics Loss: %.9f \
                 (%.9f) \t Data Loss: %.9f (%.9f) \t BC \
                 Loss: %.9f (%.9f)\n",
                iter,
                maxiters,
                loss,
                mean_loss,
                stats.physics_loss,
                mean_physics_loss,
                stats.data_loss,
                mean_data_loss,
                stats.bc_loss,
                mean_bc_loss
            )
        end

        iter += 1
        iter ≥ maxiters && break
    end

    return StatefulLuxLayer(pinn, cdev(train_state.parameters), cdev(train_state.states))
end

trained_model = train_model(xyt, target_data, xyt_bc, target_bc)
```

```
Iteration: [     1/ 50000] 	 Loss: 20.523933411 (20.523933411) 	 Physics Loss: 16.931318283 (16.931318283) 	 Data Loss: 2.007483006 (2.007483006) 	 BC Loss: 1.585133076 (1.585133076)
Iteration: [  1001/ 50000] 	 Loss: 0.017368626 (0.019241149) 	 Physics Loss: 0.000384363 (0.000523637) 	 Data Loss: 0.005318491 (0.007538575) 	 BC Loss: 0.011665772 (0.011178940)
Iteration: [  2001/ 50000] 	 Loss: 0.015431640 (0.018665662) 	 Physics Loss: 0.001248562 (0.001662039) 	 Data Loss: 0.004322520 (0.006408237) 	 BC Loss: 0.009860558 (0.010595384)
Iteration: [  3001/ 50000] 	 Loss: 0.015749741 (0.015215985) 	 Physics Loss: 0.000569926 (0.001279053) 	 Data Loss: 0.004014881 (0.004232424) 	 BC Loss: 0.011164933 (0.009704511)
Iteration: [  4001/ 50000] 	 Loss: 0.009718454 (0.008712679) 	 Physics Loss: 0.002387385 (0.003379805) 	 Data Loss: 0.003176379 (0.002104536) 	 BC Loss: 0.004154690 (0.003228336)
Iteration: [  5001/ 50000] 	 Loss: 0.004201978 (0.005531632) 	 Physics Loss: 0.001762860 (0.002109061) 	 Data Loss: 0.001470178 (0.001605706) 	 BC Loss: 0.000968940 (0.001816866)
Iteration: [  6001/ 50000] 	 Loss: 0.001058929 (0.001257410) 	 Physics Loss: 0.000292104 (0.000303334) 	 Data Loss: 0.000566745 (0.000714996) 	 BC Loss: 0.000200080 (0.000239080)
Iteration: [  7001/ 50000] 	 Loss: 0.001299282 (0.000887595) 	 Physics Loss: 0.000273338 (0.000269680) 	 Data Loss: 0.000930995 (0.000495742) 	 BC Loss: 0.000094949 (0.000122174)
Iteration: [  8001/ 50000] 	 Loss: 0.001916678 (0.001284912) 	 Physics Loss: 0.001534001 (0.000757414) 	 Data Loss: 0.000297050 (0.000406778) 	 BC Loss: 0.000085626 (0.000120719)
Iteration: [  9001/ 50000] 	 Loss: 0.001054120 (0.001594905) 	 Physics Loss: 0.000299396 (0.001037562) 	 Data Loss: 0.000645517 (0.000396183) 	 BC Loss: 0.000109207 (0.000161159)
Iteration: [ 10001/ 50000] 	 Loss: 0.000587756 (0.000591306) 	 Physics Loss: 0.000248489 (0.000239776) 	 Data Loss: 0.000290338 (0.000298280) 	 BC Loss: 0.000048929 (0.000053250)
Iteration: [ 11001/ 50000] 	 Loss: 0.000394609 (0.000378875) 	 Physics Loss: 0.000167685 (0.000071381) 	 Data Loss: 0.000182946 (0.000270383) 	 BC Loss: 0.000043979 (0.000037112)
Iteration: [ 12001/ 50000] 	 Loss: 0.000267653 (0.000351200) 	 Physics Loss: 0.000054137 (0.000067286) 	 Data Loss: 0.000172009 (0.000248348) 	 BC Loss: 0.000041506 (0.000035567)
Iteration: [ 13001/ 50000] 	 Loss: 0.000302398 (0.000331312) 	 Physics Loss: 0.000062600 (0.000070716) 	 Data Loss: 0.000206850 (0.000226218) 	 BC Loss: 0.000032949 (0.000034378)
Iteration: [ 14001/ 50000] 	 Loss: 0.000388672 (0.000332815) 	 Physics Loss: 0.000071174 (0.000069528) 	 Data Loss: 0.000285326 (0.000234454) 	 BC Loss: 0.000032172 (0.000028832)
Iteration: [ 15001/ 50000] 	 Loss: 0.000246540 (0.000294077) 	 Physics Loss: 0.000047433 (0.000061861) 	 Data Loss: 0.000169457 (0.000200651) 	 BC Loss: 0.000029650 (0.000031565)
Iteration: [ 16001/ 50000] 	 Loss: 0.000223369 (0.000292992) 	 Physics Loss: 0.000048869 (0.000061867) 	 Data Loss: 0.000142957 (0.000202374) 	 BC Loss: 0.000031544 (0.000028751)
Iteration: [ 17001/ 50000] 	 Loss: 0.000426928 (0.000298220) 	 Physics Loss: 0.000099721 (0.000067586) 	 Data Loss: 0.000303141 (0.000202573) 	 BC Loss: 0.000024066 (0.000028061)
Iteration: [ 18001/ 50000] 	 Loss: 0.000222680 (0.000288291) 	 Physics Loss: 0.000046612 (0.000063519) 	 Data Loss: 0.000141048 (0.000196650) 	 BC Loss: 0.000035020 (0.000028122)
Iteration: [ 19001/ 50000] 	 Loss: 0.000209668 (0.000279297) 	 Physics Loss: 0.000051635 (0.000056903) 	 Data Loss: 0.000139059 (0.000197520) 	 BC Loss: 0.000018975 (0.000024874)
Iteration: [ 20001/ 50000] 	 Loss: 0.000311170 (0.000250610) 	 Physics Loss: 0.000061830 (0.000047150) 	 Data Loss: 0.000229489 (0.000180880) 	 BC Loss: 0.000019851 (0.000022580)
Iteration: [ 21001/ 50000] 	 Loss: 0.000297770 (0.000249211) 	 Physics Loss: 0.000054397 (0.000053664) 	 Data Loss: 0.000219520 (0.000171688) 	 BC Loss: 0.000023853 (0.000023859)
Iteration: [ 22001/ 50000] 	 Loss: 0.000167840 (0.000244703) 	 Physics Loss: 0.000030988 (0.000052862) 	 Data Loss: 0.000109593 (0.000168477) 	 BC Loss: 0.000027260 (0.000023364)
Iteration: [ 23001/ 50000] 	 Loss: 0.000228566 (0.000250646) 	 Physics Loss: 0.000036658 (0.000051400) 	 Data Loss: 0.000169543 (0.000177615) 	 BC Loss: 0.000022365 (0.000021631)
Iteration: [ 24001/ 50000] 	 Loss: 0.000278686 (0.000252196) 	 Physics Loss: 0.000045813 (0.000057388) 	 Data Loss: 0.000213395 (0.000170075) 	 BC Loss: 0.000019478 (0.000024732)
Iteration: [ 25001/ 50000] 	 Loss: 0.000200878 (0.000227304) 	 Physics Loss: 0.000038822 (0.000037913) 	 Data Loss: 0.000143345 (0.000168155) 	 BC Loss: 0.000018711 (0.000021235)
Iteration: [ 26001/ 50000] 	 Loss: 0.000228508 (0.000248687) 	 Physics Loss: 0.000051800 (0.000061216) 	 Data Loss: 0.000154025 (0.000164366) 	 BC Loss: 0.000022684 (0.000023105)
Iteration: [ 27001/ 50000] 	 Loss: 0.000232940 (0.000243647) 	 Physics Loss: 0.000052619 (0.000055690) 	 Data Loss: 0.000152978 (0.000165052) 	 BC Loss: 0.000027343 (0.000022905)
Iteration: [ 28001/ 50000] 	 Loss: 0.000235474 (0.000219167) 	 Physics Loss: 0.000063488 (0.000039409) 	 Data Loss: 0.000152773 (0.000158231) 	 BC Loss: 0.000019213 (0.000021527)
Iteration: [ 29001/ 50000] 	 Loss: 0.000210311 (0.000242146) 	 Physics Loss: 0.000039242 (0.000061763) 	 Data Loss: 0.000141453 (0.000159045) 	 BC Loss: 0.000029616 (0.000021338)
Iteration: [ 30001/ 50000] 	 Loss: 0.000211252 (0.000226990) 	 Physics Loss: 0.000029903 (0.000044777) 	 Data Loss: 0.000156289 (0.000161345) 	 BC Loss: 0.000025059 (0.000020867)
Iteration: [ 31001/ 50000] 	 Loss: 0.000268564 (0.000220705) 	 Physics Loss: 0.000036047 (0.000041489) 	 Data Loss: 0.000213097 (0.000158476) 	 BC Loss: 0.000019421 (0.000020740)
Iteration: [ 32001/ 50000] 	 Loss: 0.000210392 (0.000210199) 	 Physics Loss: 0.000042371 (0.000037728) 	 Data Loss: 0.000148003 (0.000152325) 	 BC Loss: 0.000020017 (0.000020146)
Iteration: [ 33001/ 50000] 	 Loss: 0.000186483 (0.000208469) 	 Physics Loss: 0.000030989 (0.000038702) 	 Data Loss: 0.000134987 (0.000149162) 	 BC Loss: 0.000020506 (0.000020605)
Iteration: [ 34001/ 50000] 	 Loss: 0.000189162 (0.000196416) 	 Physics Loss: 0.000028461 (0.000032049) 	 Data Loss: 0.000141935 (0.000145094) 	 BC Loss: 0.000018765 (0.000019273)
Iteration: [ 35001/ 50000] 	 Loss: 0.000139575 (0.000222471) 	 Physics Loss: 0.000021543 (0.000055158) 	 Data Loss: 0.000096809 (0.000147475) 	 BC Loss: 0.000021223 (0.000019838)
Iteration: [ 36001/ 50000] 	 Loss: 0.000159848 (0.000198504) 	 Physics Loss: 0.000026962 (0.000033059) 	 Data Loss: 0.000114107 (0.000146700) 	 BC Loss: 0.000018778 (0.000018745)
Iteration: [ 37001/ 50000] 	 Loss: 0.000308017 (0.000193698) 	 Physics Loss: 0.000067829 (0.000029073) 	 Data Loss: 0.000221503 (0.000145455) 	 BC Loss: 0.000018684 (0.000019170)
Iteration: [ 38001/ 50000] 	 Loss: 0.000242954 (0.000204941) 	 Physics Loss: 0.000029910 (0.000036371) 	 Data Loss: 0.000190621 (0.000151124) 	 BC Loss: 0.000022422 (0.000017445)
Iteration: [ 39001/ 50000] 	 Loss: 0.000174656 (0.000202088) 	 Physics Loss: 0.000034063 (0.000037498) 	 Data Loss: 0.000124243 (0.000145799) 	 BC Loss: 0.000016350 (0.000018792)
Iteration: [ 40001/ 50000] 	 Loss: 0.000171005 (0.000198428) 	 Physics Loss: 0.000025078 (0.000034145) 	 Data Loss: 0.000123758 (0.000144448) 	 BC Loss: 0.000022169 (0.000019835)
Iteration: [ 41001/ 50000] 	 Loss: 0.000161456 (0.000195863) 	 Physics Loss: 0.000022760 (0.000030647) 	 Data Loss: 0.000119813 (0.000146689) 	 BC Loss: 0.000018883 (0.000018527)
Iteration: [ 42001/ 50000] 	 Loss: 0.000177091 (0.000191616) 	 Physics Loss: 0.000025276 (0.000027312) 	 Data Loss: 0.000136115 (0.000145421) 	 BC Loss: 0.000015700 (0.000018882)
Iteration: [ 43001/ 50000] 	 Loss: 0.000197145 (0.000193853) 	 Physics Loss: 0.000031614 (0.000029231) 	 Data Loss: 0.000149533 (0.000144585) 	 BC Loss: 0.000015999 (0.000020037)
Iteration: [ 44001/ 50000] 	 Loss: 0.000171289 (0.000187831) 	 Physics Loss: 0.000011508 (0.000025426) 	 Data Loss: 0.000145822 (0.000142353) 	 BC Loss: 0.000013958 (0.000020052)
Iteration: [ 45001/ 50000] 	 Loss: 0.000272469 (0.000209430) 	 Physics Loss: 0.000042653 (0.000036558) 	 Data Loss: 0.000202507 (0.000151368) 	 BC Loss: 0.000027309 (0.000021504)
Iteration: [ 46001/ 50000] 	 Loss: 0.000204631 (0.000192588) 	 Physics Loss: 0.000027942 (0.000036314) 	 Data Loss: 0.000158345 (0.000138282) 	 BC Loss: 0.000018344 (0.000017992)
Iteration: [ 47001/ 50000] 	 Loss: 0.000182035 (0.000194089) 	 Physics Loss: 0.000027839 (0.000035759) 	 Data Loss: 0.000137158 (0.000139560) 	 BC Loss: 0.000017038 (0.000018769)
Iteration: [ 48001/ 50000] 	 Loss: 0.000189863 (0.000186299) 	 Physics Loss: 0.000030324 (0.000029798) 	 Data Loss: 0.000141614 (0.000137973) 	 BC Loss: 0.000017925 (0.000018528)
Iteration: [ 49001/ 50000] 	 Loss: 0.000171046 (0.000195039) 	 Physics Loss: 0.000028142 (0.000035482) 	 Data Loss: 0.000121497 (0.000141365) 	 BC Loss: 0.000021407 (0.000018192)

```

## Visualizing the Results {#Visualizing-the-Results}

```julia
ts, xs, ys = 0.0f0:0.05f0:2.0f0, 0.0f0:0.02f0:2.0f0, 0.0f0:0.02f0:2.0f0
grid = stack([[elem...] for elem in vec(collect(Iterators.product(xs, ys, ts)))])

u_real = reshape(analytical_solution(grid), length(xs), length(ys), length(ts))

grid_normalized = (grid .- minimum(grid)) ./ (maximum(grid) .- minimum(grid))
u_pred = reshape(trained_model(grid_normalized), length(xs), length(ys), length(ts))
u_pred = u_pred .* (max_pde_val - min_pde_val) .+ min_pde_val

begin
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel="x", ylabel="y")
    errs = [abs.(u_pred[:, :, i] .- u_real[:, :, i]) for i in 1:length(ts)]
    Colorbar(fig[1, 2]; limits=extrema(stack(errs)))

    CairoMakie.record(fig, "pinn_nested_ad.gif", 1:length(ts); framerate=10) do i
        ax.title = "Abs. Predictor Error | Time: $(ts[i])"
        err = errs[i]
        contour!(ax, xs, ys, err; levels=10, linewidth=2)
        heatmap!(ax, xs, ys, err)
        return fig
    end

    fig
end
```

## Appendix {#Appendix}

```julia
using InteractiveUtils
InteractiveUtils.versioninfo()

if @isdefined(MLDataDevices)
    if @isdefined(CUDA) && MLDataDevices.functional(CUDADevice)
        println()
        CUDA.versioninfo()
    end

    if @isdefined(AMDGPU) && MLDataDevices.functional(AMDGPUDevice)
        println()
        AMDGPU.versioninfo()
    end
end

```

```
Julia Version 1.12.6
Commit 15346901f00 (2026-04-09 19:20 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 4 × Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, icelake-server)
  GC: Built with stock GC
Threads: 4 default, 1 interactive, 4 GC (on 4 virtual cores)
Environment:
  JULIA_DEBUG = Literate
  LD_LIBRARY_PATH = 
  JULIA_NUM_THREADS = 4
  JULIA_CPU_HARD_MEMORY_LIMIT = 100%
  JULIA_PKG_PRECOMPILE_AUTO = 0

```

***

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
