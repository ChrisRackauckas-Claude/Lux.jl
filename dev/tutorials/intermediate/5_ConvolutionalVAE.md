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
Epoch 1, Iter 39, Loss: 24867.3398438, Throughput: 4.338251 im/s
Epoch 1, Train Loss: 39527.9492188, Time: 1150.9668s, Throughput: 4.337223 im/s
Epoch 2, Iter 39, Loss: 18603.2714844, Throughput: 82.010705 im/s
Epoch 2, Train Loss: 20140.7089844, Time: 60.8703s, Throughput: 82.010479 im/s
Epoch 3, Iter 39, Loss: 16156.8310547, Throughput: 82.151185 im/s
Epoch 3, Train Loss: 16672.7187500, Time: 60.7662s, Throughput: 82.150941 im/s
Epoch 4, Iter 39, Loss: 15312.7949219, Throughput: 82.156871 im/s
Epoch 4, Train Loss: 15240.0722656, Time: 60.7620s, Throughput: 82.156657 im/s
Epoch 5, Iter 39, Loss: 13307.9902344, Throughput: 82.200118 im/s
Epoch 5, Train Loss: 14210.9707031, Time: 60.7300s, Throughput: 82.199874 im/s
Epoch 6, Iter 39, Loss: 12745.3818359, Throughput: 82.218152 im/s
Epoch 6, Train Loss: 13540.4902344, Time: 60.7167s, Throughput: 82.217902 im/s
Epoch 7, Iter 39, Loss: 13443.1601562, Throughput: 82.058200 im/s
Epoch 7, Train Loss: 12982.3476562, Time: 60.8355s, Throughput: 82.057318 im/s
Epoch 8, Iter 39, Loss: 12379.7363281, Throughput: 81.934590 im/s
Epoch 8, Train Loss: 12672.6923828, Time: 60.9268s, Throughput: 81.934354 im/s
Epoch 9, Iter 39, Loss: 12231.3515625, Throughput: 81.753535 im/s
Epoch 9, Train Loss: 12315.6806641, Time: 61.0618s, Throughput: 81.753279 im/s
Epoch 10, Iter 39, Loss: 12390.0351562, Throughput: 81.845760 im/s
Epoch 10, Train Loss: 12141.6259766, Time: 60.9930s, Throughput: 81.845471 im/s
Epoch 11, Iter 39, Loss: 11703.4472656, Throughput: 80.987459 im/s
Epoch 11, Train Loss: 11822.4550781, Time: 61.6393s, Throughput: 80.987239 im/s
Epoch 12, Iter 39, Loss: 12628.2451172, Throughput: 82.042141 im/s
Epoch 12, Train Loss: 11616.2373047, Time: 60.8470s, Throughput: 82.041886 im/s
Epoch 13, Iter 39, Loss: 11245.9091797, Throughput: 81.993192 im/s
Epoch 13, Train Loss: 11455.7861328, Time: 60.8833s, Throughput: 81.992959 im/s
Epoch 14, Iter 39, Loss: 12150.2626953, Throughput: 82.036765 im/s
Epoch 14, Train Loss: 11325.6777344, Time: 60.8510s, Throughput: 82.036497 im/s
Epoch 15, Iter 39, Loss: 10334.0791016, Throughput: 81.755338 im/s
Epoch 15, Train Loss: 11193.4980469, Time: 61.0604s, Throughput: 81.755049 im/s
Epoch 16, Iter 39, Loss: 10893.0332031, Throughput: 82.158792 im/s
Epoch 16, Train Loss: 10999.0244141, Time: 60.7606s, Throughput: 82.158514 im/s
Epoch 17, Iter 39, Loss: 10512.2539062, Throughput: 82.141557 im/s
Epoch 17, Train Loss: 10968.2968750, Time: 60.7733s, Throughput: 82.141278 im/s
Epoch 18, Iter 39, Loss: 11233.2011719, Throughput: 82.329535 im/s
Epoch 18, Train Loss: 10885.3388672, Time: 60.6346s, Throughput: 82.329279 im/s
Epoch 19, Iter 39, Loss: 10785.4179688, Throughput: 82.231253 im/s
Epoch 19, Train Loss: 10819.8916016, Time: 60.7070s, Throughput: 82.230999 im/s
Epoch 20, Iter 39, Loss: 10375.1083984, Throughput: 82.073763 im/s
Epoch 20, Train Loss: 10645.8330078, Time: 60.8235s, Throughput: 82.073515 im/s
Epoch 21, Iter 39, Loss: 10839.4453125, Throughput: 82.440626 im/s
Epoch 21, Train Loss: 10638.0361328, Time: 60.5529s, Throughput: 82.440368 im/s
Epoch 22, Iter 39, Loss: 10055.0478516, Throughput: 82.245280 im/s
Epoch 22, Train Loss: 10610.3300781, Time: 60.6967s, Throughput: 82.245017 im/s
Epoch 23, Iter 39, Loss: 10807.0341797, Throughput: 82.128883 im/s
Epoch 23, Train Loss: 10470.9550781, Time: 60.7827s, Throughput: 82.128609 im/s
Epoch 24, Iter 39, Loss: 10837.5527344, Throughput: 82.226363 im/s
Epoch 24, Train Loss: 10401.1201172, Time: 60.7106s, Throughput: 82.226130 im/s
Epoch 25, Iter 39, Loss: 9854.8496094, Throughput: 82.298053 im/s
Epoch 25, Train Loss: 10222.3681641, Time: 60.6578s, Throughput: 82.297753 im/s
Epoch 26, Iter 39, Loss: 10688.7578125, Throughput: 82.379046 im/s
Epoch 26, Train Loss: 10286.4853516, Time: 60.5981s, Throughput: 82.378796 im/s
Epoch 27, Iter 39, Loss: 10509.1376953, Throughput: 82.444534 im/s
Epoch 27, Train Loss: 10224.8994141, Time: 60.5500s, Throughput: 82.444264 im/s
Epoch 28, Iter 39, Loss: 10237.4052734, Throughput: 82.393990 im/s
Epoch 28, Train Loss: 10145.2675781, Time: 60.5871s, Throughput: 82.393713 im/s
Epoch 29, Iter 39, Loss: 11052.7402344, Throughput: 82.351941 im/s
Epoch 29, Train Loss: 10033.5673828, Time: 60.6180s, Throughput: 82.351709 im/s
Epoch 30, Iter 39, Loss: 9999.3945312, Throughput: 82.171703 im/s
Epoch 30, Train Loss: 10050.7167969, Time: 60.7510s, Throughput: 82.171452 im/s
Epoch 31, Iter 39, Loss: 10634.2587891, Throughput: 82.245497 im/s
Epoch 31, Train Loss: 10022.6728516, Time: 60.6965s, Throughput: 82.245255 im/s
Epoch 32, Iter 39, Loss: 9773.5166016, Throughput: 82.143636 im/s
Epoch 32, Train Loss: 9920.6796875, Time: 60.7718s, Throughput: 82.143402 im/s
Epoch 33, Iter 39, Loss: 9586.0820312, Throughput: 82.279379 im/s
Epoch 33, Train Loss: 9934.7109375, Time: 60.6716s, Throughput: 82.279090 im/s
Epoch 34, Iter 39, Loss: 9951.5371094, Throughput: 82.230049 im/s
Epoch 34, Train Loss: 9831.6181641, Time: 60.7079s, Throughput: 82.229822 im/s
Epoch 35, Iter 39, Loss: 10205.5000000, Throughput: 82.258091 im/s
Epoch 35, Train Loss: 9833.2929688, Time: 60.6873s, Throughput: 82.257801 im/s
Epoch 36, Iter 39, Loss: 10303.5058594, Throughput: 82.400950 im/s
Epoch 36, Train Loss: 9727.0654297, Time: 60.5820s, Throughput: 82.400710 im/s
Epoch 37, Iter 39, Loss: 9878.4746094, Throughput: 82.046122 im/s
Epoch 37, Train Loss: 9687.5068359, Time: 60.8440s, Throughput: 82.045872 im/s
Epoch 38, Iter 39, Loss: 10978.2304688, Throughput: 82.283779 im/s
Epoch 38, Train Loss: 9687.9736328, Time: 60.6683s, Throughput: 82.283502 im/s
Epoch 39, Iter 39, Loss: 10393.1035156, Throughput: 81.880721 im/s
Epoch 39, Train Loss: 9738.7636719, Time: 60.9669s, Throughput: 81.880456 im/s
Epoch 40, Iter 39, Loss: 9065.4638672, Throughput: 82.138935 im/s
Epoch 40, Train Loss: 9700.8974609, Time: 60.7753s, Throughput: 82.138670 im/s
Epoch 41, Iter 39, Loss: 9609.4921875, Throughput: 81.933990 im/s
Epoch 41, Train Loss: 9632.5937500, Time: 60.9273s, Throughput: 81.933684 im/s
Epoch 42, Iter 39, Loss: 9360.9814453, Throughput: 82.193252 im/s
Epoch 42, Train Loss: 9578.4277344, Time: 60.7351s, Throughput: 82.193014 im/s
Epoch 43, Iter 39, Loss: 9586.2675781, Throughput: 82.020430 im/s
Epoch 43, Train Loss: 9504.9277344, Time: 60.8631s, Throughput: 82.020197 im/s
Epoch 44, Iter 39, Loss: 9042.9082031, Throughput: 82.127859 im/s
Epoch 44, Train Loss: 9524.1621094, Time: 60.7835s, Throughput: 82.127615 im/s
Epoch 45, Iter 39, Loss: 9650.9160156, Throughput: 82.191390 im/s
Epoch 45, Train Loss: 9486.2978516, Time: 60.7365s, Throughput: 82.191169 im/s
Epoch 46, Iter 39, Loss: 9088.2431641, Throughput: 82.397578 im/s
Epoch 46, Train Loss: 9470.3906250, Time: 60.5845s, Throughput: 82.397347 im/s
Epoch 47, Iter 39, Loss: 9109.1718750, Throughput: 82.134513 im/s
Epoch 47, Train Loss: 9462.7451172, Time: 60.7785s, Throughput: 82.134271 im/s
Epoch 48, Iter 39, Loss: 9129.9492188, Throughput: 82.293641 im/s
Epoch 48, Train Loss: 9450.2763672, Time: 60.6610s, Throughput: 82.293400 im/s
Epoch 49, Iter 39, Loss: 9101.7607422, Throughput: 82.371990 im/s
Epoch 49, Train Loss: 9299.9804688, Time: 60.6033s, Throughput: 82.371744 im/s
Epoch 50, Iter 39, Loss: 8994.9042969, Throughput: 82.464518 im/s
Epoch 50, Train Loss: 9320.6982422, Time: 60.5353s, Throughput: 82.464279 im/s

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
