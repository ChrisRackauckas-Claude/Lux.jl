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
Epoch 1, Iter 39, Loss: 24350.8476562, Throughput: 4.312593 im/s
Epoch 1, Train Loss: 39673.3593750, Time: 1157.8282s, Throughput: 4.311521 im/s
Epoch 2, Iter 39, Loss: 18166.2304688, Throughput: 88.913283 im/s
Epoch 2, Train Loss: 20279.2636719, Time: 56.1450s, Throughput: 88.912613 im/s
Epoch 3, Iter 39, Loss: 15445.3095703, Throughput: 89.190483 im/s
Epoch 3, Train Loss: 16586.0820312, Time: 55.9703s, Throughput: 89.190169 im/s
Epoch 4, Iter 39, Loss: 15212.3808594, Throughput: 88.913343 im/s
Epoch 4, Train Loss: 15131.0351562, Time: 56.1447s, Throughput: 88.913049 im/s
Epoch 5, Iter 39, Loss: 14017.4521484, Throughput: 89.022883 im/s
Epoch 5, Train Loss: 14108.6523438, Time: 56.0757s, Throughput: 89.022593 im/s
Epoch 6, Iter 39, Loss: 13293.0878906, Throughput: 89.989462 im/s
Epoch 6, Train Loss: 13469.7470703, Time: 55.4734s, Throughput: 89.989146 im/s
Epoch 7, Iter 39, Loss: 13091.8916016, Throughput: 90.569255 im/s
Epoch 7, Train Loss: 12946.4843750, Time: 55.1182s, Throughput: 90.568953 im/s
Epoch 8, Iter 39, Loss: 12943.0390625, Throughput: 89.470907 im/s
Epoch 8, Train Loss: 12562.3662109, Time: 55.7949s, Throughput: 89.470607 im/s
Epoch 9, Iter 39, Loss: 11990.1552734, Throughput: 89.558148 im/s
Epoch 9, Train Loss: 12335.0195312, Time: 55.7405s, Throughput: 89.557793 im/s
Epoch 10, Iter 39, Loss: 11225.9453125, Throughput: 90.141291 im/s
Epoch 10, Train Loss: 12055.6738281, Time: 55.3799s, Throughput: 90.140965 im/s
Epoch 11, Iter 39, Loss: 12239.2177734, Throughput: 90.563823 im/s
Epoch 11, Train Loss: 11794.5390625, Time: 55.1215s, Throughput: 90.563498 im/s
Epoch 12, Iter 39, Loss: 12083.8779297, Throughput: 90.756296 im/s
Epoch 12, Train Loss: 11678.9228516, Time: 55.0046s, Throughput: 90.756003 im/s
Epoch 13, Iter 39, Loss: 11254.1152344, Throughput: 91.548074 im/s
Epoch 13, Train Loss: 11457.0683594, Time: 54.5289s, Throughput: 91.547771 im/s
Epoch 14, Iter 39, Loss: 11320.1162109, Throughput: 91.127810 im/s
Epoch 14, Train Loss: 11333.8525391, Time: 54.7804s, Throughput: 91.127515 im/s
Epoch 15, Iter 39, Loss: 11442.9941406, Throughput: 91.750783 im/s
Epoch 15, Train Loss: 11156.4384766, Time: 54.4085s, Throughput: 91.750448 im/s
Epoch 16, Iter 39, Loss: 11544.3886719, Throughput: 91.218370 im/s
Epoch 16, Train Loss: 11112.7939453, Time: 54.7260s, Throughput: 91.218013 im/s
Epoch 17, Iter 39, Loss: 11021.7509766, Throughput: 91.699749 im/s
Epoch 17, Train Loss: 10903.6425781, Time: 54.4387s, Throughput: 91.699443 im/s
Epoch 18, Iter 39, Loss: 10923.1445312, Throughput: 91.434272 im/s
Epoch 18, Train Loss: 10873.0322266, Time: 54.5968s, Throughput: 91.433956 im/s
Epoch 19, Iter 39, Loss: 10487.4082031, Throughput: 91.418835 im/s
Epoch 19, Train Loss: 10771.9277344, Time: 54.6060s, Throughput: 91.418530 im/s
Epoch 20, Iter 39, Loss: 10645.0019531, Throughput: 92.114944 im/s
Epoch 20, Train Loss: 10697.5214844, Time: 54.1934s, Throughput: 92.114621 im/s
Epoch 21, Iter 39, Loss: 11136.0410156, Throughput: 91.460070 im/s
Epoch 21, Train Loss: 10636.7587891, Time: 54.5814s, Throughput: 91.459770 im/s
Epoch 22, Iter 39, Loss: 10522.6582031, Throughput: 91.771248 im/s
Epoch 22, Train Loss: 10573.5605469, Time: 54.3963s, Throughput: 91.770937 im/s
Epoch 23, Iter 39, Loss: 10053.2128906, Throughput: 90.660358 im/s
Epoch 23, Train Loss: 10432.5566406, Time: 55.0628s, Throughput: 90.660038 im/s
Epoch 24, Iter 39, Loss: 10098.4335938, Throughput: 91.325212 im/s
Epoch 24, Train Loss: 10362.7460938, Time: 54.6620s, Throughput: 91.324898 im/s
Epoch 25, Iter 39, Loss: 10489.7851562, Throughput: 91.420722 im/s
Epoch 25, Train Loss: 10276.1240234, Time: 54.6049s, Throughput: 91.420439 im/s
Epoch 26, Iter 39, Loss: 9789.2802734, Throughput: 91.642776 im/s
Epoch 26, Train Loss: 10246.4580078, Time: 54.4726s, Throughput: 91.642478 im/s
Epoch 27, Iter 39, Loss: 9760.4824219, Throughput: 91.199119 im/s
Epoch 27, Train Loss: 10160.5917969, Time: 54.7375s, Throughput: 91.198827 im/s
Epoch 28, Iter 39, Loss: 10108.8837891, Throughput: 92.259986 im/s
Epoch 28, Train Loss: 10052.9980469, Time: 54.1081s, Throughput: 92.259689 im/s
Epoch 29, Iter 39, Loss: 9884.3085938, Throughput: 91.450047 im/s
Epoch 29, Train Loss: 9990.2607422, Time: 54.5874s, Throughput: 91.449741 im/s
Epoch 30, Iter 39, Loss: 9803.5917969, Throughput: 92.537691 im/s
Epoch 30, Train Loss: 10069.5820312, Time: 53.9458s, Throughput: 92.537366 im/s
Epoch 31, Iter 39, Loss: 9927.2148438, Throughput: 90.859478 im/s
Epoch 31, Train Loss: 9995.7695312, Time: 54.9422s, Throughput: 90.859185 im/s
Epoch 32, Iter 39, Loss: 9542.4628906, Throughput: 92.090070 im/s
Epoch 32, Train Loss: 9887.9931641, Time: 54.2080s, Throughput: 92.089754 im/s
Epoch 33, Iter 39, Loss: 10142.5917969, Throughput: 91.147798 im/s
Epoch 33, Train Loss: 9872.3388672, Time: 54.7684s, Throughput: 91.147523 im/s
Epoch 34, Iter 39, Loss: 9574.2060547, Throughput: 91.778910 im/s
Epoch 34, Train Loss: 9870.4111328, Time: 54.3918s, Throughput: 91.778606 im/s
Epoch 35, Iter 39, Loss: 9302.5351562, Throughput: 91.615987 im/s
Epoch 35, Train Loss: 9768.3017578, Time: 54.4885s, Throughput: 91.615639 im/s
Epoch 36, Iter 39, Loss: 9697.2226562, Throughput: 91.146943 im/s
Epoch 36, Train Loss: 9821.7236328, Time: 54.7689s, Throughput: 91.146639 im/s
Epoch 37, Iter 39, Loss: 10167.8818359, Throughput: 90.124013 im/s
Epoch 37, Train Loss: 9826.5166016, Time: 55.3905s, Throughput: 90.123717 im/s
Epoch 38, Iter 39, Loss: 9330.0566406, Throughput: 91.043523 im/s
Epoch 38, Train Loss: 9700.0849609, Time: 54.8311s, Throughput: 91.043204 im/s
Epoch 39, Iter 39, Loss: 9981.2890625, Throughput: 90.942089 im/s
Epoch 39, Train Loss: 9680.5820312, Time: 54.8923s, Throughput: 90.941765 im/s
Epoch 40, Iter 39, Loss: 9861.4501953, Throughput: 90.978419 im/s
Epoch 40, Train Loss: 9605.5908203, Time: 54.8703s, Throughput: 90.978102 im/s
Epoch 41, Iter 39, Loss: 9779.1230469, Throughput: 91.264973 im/s
Epoch 41, Train Loss: 9695.5986328, Time: 54.6980s, Throughput: 91.264698 im/s
Epoch 42, Iter 39, Loss: 10046.9277344, Throughput: 90.599297 im/s
Epoch 42, Train Loss: 9541.9042969, Time: 55.0999s, Throughput: 90.599005 im/s
Epoch 43, Iter 39, Loss: 9264.5019531, Throughput: 91.508153 im/s
Epoch 43, Train Loss: 9504.5595703, Time: 54.5527s, Throughput: 91.507857 im/s
Epoch 44, Iter 39, Loss: 9717.9921875, Throughput: 90.979327 im/s
Epoch 44, Train Loss: 9541.0957031, Time: 54.8698s, Throughput: 90.978995 im/s
Epoch 45, Iter 39, Loss: 9810.7939453, Throughput: 91.343831 im/s
Epoch 45, Train Loss: 9525.8105469, Time: 54.6509s, Throughput: 91.343500 im/s
Epoch 46, Iter 39, Loss: 9306.2500000, Throughput: 91.089477 im/s
Epoch 46, Train Loss: 9447.4794922, Time: 54.8035s, Throughput: 91.089156 im/s
Epoch 47, Iter 39, Loss: 9091.5898438, Throughput: 91.533991 im/s
Epoch 47, Train Loss: 9421.0263672, Time: 54.5373s, Throughput: 91.533678 im/s
Epoch 48, Iter 39, Loss: 9445.6533203, Throughput: 92.014788 im/s
Epoch 48, Train Loss: 9339.1005859, Time: 54.2523s, Throughput: 92.014500 im/s
Epoch 49, Iter 39, Loss: 9054.8515625, Throughput: 92.154511 im/s
Epoch 49, Train Loss: 9382.4365234, Time: 54.1701s, Throughput: 92.154166 im/s
Epoch 50, Iter 39, Loss: 9433.7929688, Throughput: 90.505802 im/s
Epoch 50, Train Loss: 9310.6386719, Time: 55.1569s, Throughput: 90.505507 im/s

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
  CPU: 4 × AMD EPYC 9V74 80-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, znver4)
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
