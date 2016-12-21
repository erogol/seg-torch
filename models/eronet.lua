--
--  Copyright (c) 2016, Fedor Chervinskii
--

require 'nn'
require 'torch'   -- torch
torch.setdefaulttensortype('torch.FloatTensor')


local classes = opt.classes
local classWeights = opt.classWeights

print '==> construct model'

local function add_block(cont,n_conv,sizes,wid,str,pad)
    local wid = wid or 3
    local str = str or 1
    local pad = pad or 1
    for i=1,n_conv do
        cont:add(cudnn.SpatialConvolution(sizes[i],sizes[i+1],wid,wid,str,str,pad,pad))
        -- cont:add(nn.SpatialBatchNormalization(sizes[i+1]))
        cont:add(nn.ELU())
    end
    return cont
end


-- conv_sizes = {3,64,64,128,128,256,256,256,512,512,512,512,512,512}
conv_sizes = {3,32,32,64,64,128,128,128,256,256,256,256,256,256}

encoder = nn.Sequential()
pool = {}

counter = 1
for i=1,2 do
    sizes = {conv_sizes[counter],conv_sizes[counter+1],conv_sizes[counter+2]}
    encoder = add_block(encoder,2,sizes)
    counter = counter + 2
    pool[i] = nn.SpatialMaxPooling(2,2,2,2)
    encoder:add(pool[i])
end
for i=3,5 do
    sizes = {conv_sizes[counter],conv_sizes[counter+1],conv_sizes[counter+2],conv_sizes[counter+3]}
    encoder = add_block(encoder,3,sizes)
    counter = counter + 3
    pool[i] = nn.SpatialMaxPooling(2,2,2,2)
    encoder:add(pool[i])
end

decoder = nn.Sequential()

counter = #conv_sizes
for i=5,3,-1 do
    decoder:add(nn.SpatialMaxUnpooling(pool[i]))
    sizes = {conv_sizes[counter],conv_sizes[counter-1],conv_sizes[counter-2],conv_sizes[counter-3]}
    decoder = add_block(decoder,3,sizes)
    counter = counter - 3
end
for i=2,1,-1 do
    decoder:add(nn.SpatialMaxUnpooling(pool[i]))
    sizes = {conv_sizes[counter],conv_sizes[counter-1],conv_sizes[counter-2]}
    counter = counter - 2
    decoder = add_block(decoder,i,sizes)
end
decoder:add(nn.SpatialConvolution(conv_sizes[2],#classes,3,3,1,1,1))
print(" | ==> Last layer:"..conv_sizes[2].." --> "..#classes)

net = nn.Sequential()
net:add(encoder)
net:add(decoder)
net:cuda()

-- optimize model memory usage
print(' | ==> optnet optimization...')
local optnet = require 'optnet'
local sampleInput = torch.zeros(2,3,opt.imHeight,opt.imWidth):cuda()
optnet.optimizeMemory(net, sampleInput, {inplace = true, mode = 'training'})

-- init model
local function ConvInit(name)
    for k,v in pairs(net:findModules(name)) do
        local n = v.kW*v.kH*v.nOutputPlane
        v.weight:normal(0,math.sqrt(2/n))
        if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
        else
            v.bias:zero()
        end
    end
end
ConvInit('cudnn.SpatialConvolution')
ConvInit('nn.SpatialConvolution')

-- set loss function
local loss
if opt.classWeights then
    print(' | ==> setting loss layer with class weights ')
    -- print(classWeights)
    loss = cudnn.SpatialCrossEntropyCriterion(classWeights)
else
    loss = cudnn.SpatialCrossEntropyCriterion(classWeights)
end
loss:cuda()

---------------------------------------------------------------------
-- sample model run
print(" | ==> sample model run")
local rnd_input = torch.rand(4, 3, opt.imHeight, opt.imWidth):cuda()
local output = net:forward(rnd_input)
print(" | | ==> SegNet model output size (masks will be resized to these values) -- ")
print(" | | | ==> width: ".. output:size(4))
print(" | | | ==> height: ".. output:size(3))

assert(opt.maskHeight == output:size(3))
assert(opt.maskWidth == output:size(4))


return { model = net, loss = loss}
