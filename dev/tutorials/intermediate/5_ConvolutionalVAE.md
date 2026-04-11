---
url: /dev/tutorials/intermediate/5_ConvolutionalVAE.md
---
# Convolutional VAE for MNIST {#Convolutional-VAE-Tutorial}

Convolutional variational autoencoder (CVAE) implementation in MLX using MNIST. This is based on the [CVAE implementation in MLX](https://github.com/ml-explore/mlx-examples/blob/main/cvae/).

```julia
using Lux,
    Reactant,
    MLDatasets,
    Random,
    Statistics,
    Enzyme,
    MLUtils,
    DataAugmentation,
    ConcreteStructs,
    OneHotArrays,
    ImageShow,
    Images,
    Printf,
    Optimisers

const xdev = reactant_device(; force=true)
const cdev = cpu_device()

const IN_VSCODE = isdefined(Main, :VSCodeServer)
```

```
false
```

## Model Definition {#Model-Definition}

First we will define the encoder.It maps the input to a normal distribution in latent space and sample a latent vector from that distribution.

```julia
function cvae_encoder(
    rng=Random.default_rng();
    num_latent_dims::Int,
    image_shape::Dims{3},
    max_num_filters::Int,
)
    flattened_dim = prod(image_shape[1:2] .÷ 8) * max_num_filters
    return @compact(;
        embed=Chain(
            Chain(
                Conv((3, 3), image_shape[3] => max_num_filters ÷ 4; stride=2, pad=1),
                BatchNorm(max_num_filters ÷ 4, leakyrelu),
            ),
            Chain(
                Conv((3, 3), max_num_filters ÷ 4 => max_num_filters ÷ 2; stride=2, pad=1),
                BatchNorm(max_num_filters ÷ 2, leakyrelu),
            ),
            Chain(
                Conv((3, 3), max_num_filters ÷ 2 => max_num_filters; stride=2, pad=1),
                BatchNorm(max_num_filters, leakyrelu),
            ),
            FlattenLayer(),
        ),
        proj_mu=Dense(flattened_dim, num_latent_dims; init_bias=zeros32),
        proj_log_var=Dense(flattened_dim, num_latent_dims; init_bias=zeros32),
        rng
    ) do x
        y = embed(x)

        μ = proj_mu(y)
        logσ² = proj_log_var(y)

        T = eltype(logσ²)
        logσ² = clamp.(logσ², -T(20.0f0), T(10.0f0))
        σ = exp.(logσ² .* T(0.5))

        # Generate a tensor of random values from a normal distribution
        ϵ = randn_like(Lux.replicate(rng), σ)

        # Reparameterization trick to backpropagate through sampling
        z = ϵ .* σ .+ μ

        @return z, μ, logσ²
    end
end
```

Similarly we define the decoder.

```julia
function cvae_decoder(; num_latent_dims::Int, image_shape::Dims{3}, max_num_filters::Int)
    flattened_dim = prod(image_shape[1:2] .÷ 8) * max_num_filters
    return @compact(;
        linear=Dense(num_latent_dims, flattened_dim),
        upchain=Chain(
            Chain(
                Upsample(2),
                Conv((3, 3), max_num_filters => max_num_filters ÷ 2; stride=1, pad=1),
                BatchNorm(max_num_filters ÷ 2, leakyrelu),
            ),
            Chain(
                Upsample(2),
                Conv((3, 3), max_num_filters ÷ 2 => max_num_filters ÷ 4; stride=1, pad=1),
                BatchNorm(max_num_filters ÷ 4, leakyrelu),
            ),
            Chain(
                Upsample(2),
                Conv(
                    (3, 3), max_num_filters ÷ 4 => image_shape[3], sigmoid; stride=1, pad=1
                ),
            ),
        ),
        max_num_filters
    ) do x
        y = linear(x)
        img = reshape(y, image_shape[1] ÷ 8, image_shape[2] ÷ 8, max_num_filters, :)
        @return upchain(img)
    end
end

@concrete struct CVAE <: AbstractLuxContainerLayer{(:encoder, :decoder)}
    encoder <: AbstractLuxLayer
    decoder <: AbstractLuxLayer
end

function CVAE(
    rng=Random.default_rng();
    num_latent_dims::Int,
    image_shape::Dims{3},
    max_num_filters::Int,
)
    decoder = cvae_decoder(; num_latent_dims, image_shape, max_num_filters)
    encoder = cvae_encoder(rng; num_latent_dims, image_shape, max_num_filters)
    return CVAE(encoder, decoder)
end

function (cvae::CVAE)(x, ps, st)
    (z, μ, logσ²), st_enc = cvae.encoder(x, ps.encoder, st.encoder)
    x_rec, st_dec = cvae.decoder(z, ps.decoder, st.decoder)
    return (x_rec, μ, logσ²), (; encoder=st_enc, decoder=st_dec)
end

function encode(cvae::CVAE, x, ps, st)
    (z, _, _), st_enc = cvae.encoder(x, ps.encoder, st.encoder)
    return z, (; encoder=st_enc, st.decoder)
end

function decode(cvae::CVAE, z, ps, st)
    x_rec, st_dec = cvae.decoder(z, ps.decoder, st.decoder)
    return x_rec, (; decoder=st_dec, st.encoder)
end
```

## Loading MNIST {#Loading-MNIST}

```julia
@concrete struct TensorDataset
    dataset
    transform
    total_samples::Int
end

Base.length(ds::TensorDataset) = ds.total_samples

function Base.getindex(ds::TensorDataset, idxs::Union{Vector{<:Integer},AbstractRange})
    img = Image.(eachslice(convert2image(ds.dataset, idxs); dims=3))
    return stack(parent ∘ itemdata ∘ Base.Fix1(apply, ds.transform), img)
end

function loadmnist(batchsize, image_size::Dims{2})
    # Load MNIST: Only 1500 for demonstration purposes on CI
    train_dataset = MNIST(; split=:train)
    N = parse(Bool, get(ENV, "CI", "false")) ? 5000 : length(train_dataset)

    train_transform = ScaleKeepAspect(image_size) |> ImageToTensor()
    trainset = TensorDataset(train_dataset, train_transform, N)
    trainloader = DataLoader(trainset; batchsize, shuffle=true, partial=false)

    return trainloader
end
```

## Helper Functions {#Helper-Functions}

Generate an Image Grid from a list of images

```julia
function create_image_grid(imgs::AbstractArray, grid_rows::Int, grid_cols::Int)
    total_images = grid_rows * grid_cols
    imgs = map(eachslice(imgs[:, :, :, 1:total_images]; dims=4)) do img
        cimg = if size(img, 3) == 1
            colorview(Gray, view(img, :, :, 1))
        else
            colorview(RGB, permutedims(img, (3, 1, 2)))
        end
        return cimg'
    end
    return create_image_grid(imgs, grid_rows, grid_cols)
end

function create_image_grid(images::Vector, grid_rows::Int, grid_cols::Int)
    # Check if the number of images matches the grid
    total_images = grid_rows * grid_cols
    @assert length(images) == total_images

    # Get the size of a single image (assuming all images are the same size)
    img_height, img_width = size(images[1])

    # Create a blank grid canvas
    grid_height = img_height * grid_rows
    grid_width = img_width * grid_cols
    grid_canvas = similar(images[1], grid_height, grid_width)

    # Place each image in the correct position on the canvas
    for idx in 1:total_images
        row = div(idx - 1, grid_cols) + 1
        col = mod(idx - 1, grid_cols) + 1

        start_row = (row - 1) * img_height + 1
        start_col = (col - 1) * img_width + 1

        grid_canvas[start_row:(start_row + img_height - 1), start_col:(start_col + img_width - 1)] .= images[idx]
    end

    return grid_canvas
end

function loss_function(model, ps, st, X)
    (y, μ, logσ²), st = model(X, ps, st)
    reconstruction_loss = MSELoss(; agg=sum)(y, X)
    kldiv_loss = -sum(1 .+ logσ² .- μ .^ 2 .- exp.(logσ²)) / 2
    loss = reconstruction_loss + kldiv_loss
    return loss, st, (; y, μ, logσ², reconstruction_loss, kldiv_loss)
end

function generate_images(
    model, ps, st; num_samples::Int=128, num_latent_dims::Int, decode_compiled=nothing
)
    z = get_device((ps, st))(randn(Float32, num_latent_dims, num_samples))
    if decode_compiled === nothing
        images, _ = decode(model, z, ps, Lux.testmode(st))
    else
        images, _ = decode_compiled(model, z, ps, Lux.testmode(st))
        images = cpu_device()(images)
    end
    return create_image_grid(images, 8, num_samples ÷ 8)
end

function reconstruct_images(model, ps, st, X)
    (recon, _, _), _ = model(X, ps, Lux.testmode(st))
    recon = cpu_device()(recon)
    return create_image_grid(recon, 8, size(X, ndims(X)) ÷ 8)
end
```

```
reconstruct_images (generic function with 1 method)
```

## Training the Model {#Training-the-Model}

```julia
function main(;
    batchsize=128,
    image_size=(64, 64),
    num_latent_dims=8,
    max_num_filters=64,
    seed=0,
    epochs=50,
    weight_decay=1.0e-5,
    learning_rate=1.0e-3,
    num_samples=batchsize,
)
    rng = Xoshiro()
    Random.seed!(rng, seed)

    cvae = CVAE(rng; num_latent_dims, image_shape=(image_size..., 1), max_num_filters)
    ps, st = Lux.setup(rng, cvae) |> xdev

    z = xdev(randn(Float32, num_latent_dims, num_samples))
    decode_compiled = @compile decode(cvae, z, ps, Lux.testmode(st))
    x = randn(Float32, image_size..., 1, batchsize) |> xdev
    cvae_compiled = @compile cvae(x, ps, Lux.testmode(st))

    train_dataloader = loadmnist(batchsize, image_size) |> xdev

    opt = AdamW(; eta=learning_rate, lambda=weight_decay)

    train_state = Training.TrainState(cvae, ps, st, opt)

    @printf "Total Trainable Parameters: %0.4f M\n" (Lux.parameterlength(ps) / 1.0e6)

    empty_row, model_img_full = nothing, nothing

    for epoch in 1:epochs
        loss_total = 0.0f0
        total_samples = 0

        start_time = time()
        for (i, X) in enumerate(train_dataloader)
            (_, loss, _, train_state) = Training.single_train_step!(
                AutoEnzyme(), loss_function, X, train_state; return_gradients=Val(false)
            )

            loss_total += loss
            total_samples += size(X, ndims(X))

            if i % 250 == 0 || i == length(train_dataloader)
                throughput = total_samples / (time() - start_time)
                @printf "Epoch %d, Iter %d, Loss: %.7f, Throughput: %.6f im/s\n" epoch i loss throughput
            end
        end
        total_time = time() - start_time

        train_loss = loss_total / length(train_dataloader)
        throughput = total_samples / total_time
        @printf "Epoch %d, Train Loss: %.7f, Time: %.4fs, Throughput: %.6f im/s\n" epoch train_loss total_time throughput

        if IN_VSCODE || epoch == epochs
            recon_images = reconstruct_images(
                cvae_compiled,
                train_state.parameters,
                train_state.states,
                first(train_dataloader),
            )
            gen_images = generate_images(
                cvae,
                train_state.parameters,
                train_state.states;
                num_samples,
                num_latent_dims,
                decode_compiled,
            )
            if empty_row === nothing
                empty_row = similar(gen_images, image_size[1], size(gen_images, 2))
                fill!(empty_row, 0)
            end
            model_img_full = vcat(recon_images, empty_row, gen_images)
            IN_VSCODE && display(model_img_full)
        end
    end

    return model_img_full
end

img = main()
```

```
Total Trainable Parameters: 0.1493 M
Epoch 1, Iter 39, Loss: 24198.3007812, Throughput: 3.994776 im/s
Epoch 1, Train Loss: 39687.0429688, Time: 1249.9333s, Throughput: 3.993813 im/s
Epoch 2, Iter 39, Loss: 17570.7734375, Throughput: 72.844988 im/s
Epoch 2, Train Loss: 20172.9589844, Time: 68.5293s, Throughput: 72.844785 im/s
Epoch 3, Iter 39, Loss: 16230.7109375, Throughput: 72.699194 im/s
Epoch 3, Train Loss: 16748.6289062, Time: 68.6667s, Throughput: 72.699017 im/s
Epoch 4, Iter 39, Loss: 15441.9843750, Throughput: 72.461250 im/s
Epoch 4, Train Loss: 15193.6250000, Time: 68.8922s, Throughput: 72.461058 im/s
Epoch 5, Iter 39, Loss: 13964.2460938, Throughput: 72.564893 im/s
Epoch 5, Train Loss: 14227.4667969, Time: 68.7938s, Throughput: 72.564717 im/s
Epoch 6, Iter 39, Loss: 12951.2744141, Throughput: 72.651560 im/s
Epoch 6, Train Loss: 13558.5126953, Time: 68.7117s, Throughput: 72.651378 im/s
Epoch 7, Iter 39, Loss: 12171.1718750, Throughput: 72.346900 im/s
Epoch 7, Train Loss: 13009.3281250, Time: 69.0011s, Throughput: 72.346722 im/s
Epoch 8, Iter 39, Loss: 12832.9218750, Throughput: 72.261639 im/s
Epoch 8, Train Loss: 12586.4218750, Time: 69.0825s, Throughput: 72.261444 im/s
Epoch 9, Iter 39, Loss: 12030.1494141, Throughput: 72.538832 im/s
Epoch 9, Train Loss: 12407.7294922, Time: 68.8185s, Throughput: 72.538647 im/s
Epoch 10, Iter 39, Loss: 11756.5625000, Throughput: 72.539076 im/s
Epoch 10, Train Loss: 12078.4082031, Time: 68.8182s, Throughput: 72.538915 im/s
Epoch 11, Iter 39, Loss: 11701.8251953, Throughput: 72.557020 im/s
Epoch 11, Train Loss: 11892.9746094, Time: 68.8012s, Throughput: 72.556838 im/s
Epoch 12, Iter 39, Loss: 11353.7871094, Throughput: 72.021591 im/s
Epoch 12, Train Loss: 11708.5781250, Time: 69.3127s, Throughput: 72.021425 im/s
Epoch 13, Iter 39, Loss: 11335.5253906, Throughput: 71.994263 im/s
Epoch 13, Train Loss: 11520.2597656, Time: 69.3390s, Throughput: 71.994104 im/s
Epoch 14, Iter 39, Loss: 11357.0390625, Throughput: 72.508024 im/s
Epoch 14, Train Loss: 11291.8906250, Time: 68.8477s, Throughput: 72.507837 im/s
Epoch 15, Iter 39, Loss: 11017.7011719, Throughput: 72.533205 im/s
Epoch 15, Train Loss: 11225.6533203, Time: 68.8238s, Throughput: 72.533030 im/s
Epoch 16, Iter 39, Loss: 11146.1103516, Throughput: 71.700661 im/s
Epoch 16, Train Loss: 11046.8593750, Time: 69.6230s, Throughput: 71.700466 im/s
Epoch 17, Iter 39, Loss: 10310.8935547, Throughput: 71.886323 im/s
Epoch 17, Train Loss: 10937.9912109, Time: 69.4431s, Throughput: 71.886159 im/s
Epoch 18, Iter 39, Loss: 10626.5468750, Throughput: 72.044792 im/s
Epoch 18, Train Loss: 10812.7451172, Time: 69.2904s, Throughput: 72.044614 im/s
Epoch 19, Iter 39, Loss: 10812.7246094, Throughput: 71.766351 im/s
Epoch 19, Train Loss: 10713.2578125, Time: 69.5592s, Throughput: 71.766183 im/s
Epoch 20, Iter 39, Loss: 10864.6699219, Throughput: 72.286715 im/s
Epoch 20, Train Loss: 10757.7744141, Time: 69.0585s, Throughput: 72.286564 im/s
Epoch 21, Iter 39, Loss: 10520.2314453, Throughput: 72.650335 im/s
Epoch 21, Train Loss: 10680.2255859, Time: 68.7129s, Throughput: 72.650152 im/s
Epoch 22, Iter 39, Loss: 10829.9794922, Throughput: 72.641515 im/s
Epoch 22, Train Loss: 10537.1093750, Time: 68.7212s, Throughput: 72.641355 im/s
Epoch 23, Iter 39, Loss: 11324.9550781, Throughput: 72.346019 im/s
Epoch 23, Train Loss: 10521.8417969, Time: 69.0019s, Throughput: 72.345863 im/s
Epoch 24, Iter 39, Loss: 10453.7978516, Throughput: 72.347069 im/s
Epoch 24, Train Loss: 10465.5761719, Time: 69.0009s, Throughput: 72.346891 im/s
Epoch 25, Iter 39, Loss: 10345.0146484, Throughput: 72.693019 im/s
Epoch 25, Train Loss: 10303.5507812, Time: 68.6725s, Throughput: 72.692821 im/s
Epoch 26, Iter 39, Loss: 10280.3525391, Throughput: 72.316899 im/s
Epoch 26, Train Loss: 10258.1708984, Time: 69.0297s, Throughput: 72.316718 im/s
Epoch 27, Iter 39, Loss: 10053.3154297, Throughput: 71.746063 im/s
Epoch 27, Train Loss: 10260.6064453, Time: 69.5789s, Throughput: 71.745895 im/s
Epoch 28, Iter 39, Loss: 10237.8320312, Throughput: 72.322758 im/s
Epoch 28, Train Loss: 10164.1455078, Time: 69.0241s, Throughput: 72.322603 im/s
Epoch 29, Iter 39, Loss: 10359.0693359, Throughput: 72.196149 im/s
Epoch 29, Train Loss: 10113.4667969, Time: 69.1451s, Throughput: 72.195970 im/s
Epoch 30, Iter 39, Loss: 10230.0292969, Throughput: 72.161251 im/s
Epoch 30, Train Loss: 10085.2978516, Time: 69.1786s, Throughput: 72.161087 im/s
Epoch 31, Iter 39, Loss: 9429.4082031, Throughput: 72.203983 im/s
Epoch 31, Train Loss: 10043.3193359, Time: 69.1376s, Throughput: 72.203826 im/s
Epoch 32, Iter 39, Loss: 10953.5712891, Throughput: 72.255821 im/s
Epoch 32, Train Loss: 9981.9150391, Time: 69.0880s, Throughput: 72.255645 im/s
Epoch 33, Iter 39, Loss: 10134.6269531, Throughput: 72.400588 im/s
Epoch 33, Train Loss: 9945.5644531, Time: 68.9499s, Throughput: 72.400411 im/s
Epoch 34, Iter 39, Loss: 10206.5126953, Throughput: 71.881560 im/s
Epoch 34, Train Loss: 9861.2763672, Time: 69.4477s, Throughput: 71.881395 im/s
Epoch 35, Iter 39, Loss: 10074.6054688, Throughput: 72.769229 im/s
Epoch 35, Train Loss: 9864.6816406, Time: 68.6006s, Throughput: 72.769067 im/s
Epoch 36, Iter 39, Loss: 9902.9902344, Throughput: 72.522439 im/s
Epoch 36, Train Loss: 9824.2207031, Time: 68.8340s, Throughput: 72.522277 im/s
Epoch 37, Iter 39, Loss: 9493.8583984, Throughput: 71.987415 im/s
Epoch 37, Train Loss: 9724.8486328, Time: 69.3456s, Throughput: 71.987259 im/s
Epoch 38, Iter 39, Loss: 10791.1318359, Throughput: 72.149801 im/s
Epoch 38, Train Loss: 9696.7041016, Time: 69.1895s, Throughput: 72.149643 im/s
Epoch 39, Iter 39, Loss: 9319.6757812, Throughput: 72.047181 im/s
Epoch 39, Train Loss: 9726.1005859, Time: 69.2881s, Throughput: 72.047031 im/s
Epoch 40, Iter 39, Loss: 9148.2832031, Throughput: 71.973774 im/s
Epoch 40, Train Loss: 9670.2792969, Time: 69.3588s, Throughput: 71.973614 im/s
Epoch 41, Iter 39, Loss: 9685.4970703, Throughput: 72.483751 im/s
Epoch 41, Train Loss: 9578.7343750, Time: 68.8708s, Throughput: 72.483556 im/s
Epoch 42, Iter 39, Loss: 9721.0546875, Throughput: 71.940423 im/s
Epoch 42, Train Loss: 9571.9375000, Time: 69.3909s, Throughput: 71.940258 im/s
Epoch 43, Iter 39, Loss: 9382.9873047, Throughput: 71.968749 im/s
Epoch 43, Train Loss: 9538.5869141, Time: 69.3636s, Throughput: 71.968578 im/s
Epoch 44, Iter 39, Loss: 9381.0957031, Throughput: 72.058885 im/s
Epoch 44, Train Loss: 9500.1298828, Time: 69.2768s, Throughput: 72.058713 im/s
Epoch 45, Iter 39, Loss: 9579.1972656, Throughput: 72.409945 im/s
Epoch 45, Train Loss: 9484.0625000, Time: 68.9410s, Throughput: 72.409768 im/s
Epoch 46, Iter 39, Loss: 9026.7656250, Throughput: 72.229507 im/s
Epoch 46, Train Loss: 9501.2968750, Time: 69.1132s, Throughput: 72.229347 im/s
Epoch 47, Iter 39, Loss: 9472.1054688, Throughput: 71.788067 im/s
Epoch 47, Train Loss: 9533.7890625, Time: 69.5382s, Throughput: 71.787894 im/s
Epoch 48, Iter 39, Loss: 9042.3964844, Throughput: 72.136964 im/s
Epoch 48, Train Loss: 9408.0458984, Time: 69.2019s, Throughput: 72.136793 im/s
Epoch 49, Iter 39, Loss: 9678.3261719, Throughput: 72.641009 im/s
Epoch 49, Train Loss: 9357.6416016, Time: 68.7217s, Throughput: 72.640839 im/s
Epoch 50, Iter 39, Loss: 9065.9238281, Throughput: 72.371589 im/s
Epoch 50, Train Loss: 9449.1123047, Time: 68.9775s, Throughput: 72.371413 im/s

```

***

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
Julia Version 1.12.5
Commit 5fe89b8ddc1 (2026-02-09 16:05 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 4 × AMD EPYC 7763 64-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, znver3)
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
