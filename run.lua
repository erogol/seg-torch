----------------------------------------------------------------------
-- Main script for training a model for semantic segmentation
--
-- Abhishek Chaurasia, Eugenio Culurciello
-- Sangpil Kim, Adam Paszke
-- Edited by Eren Golge
----------------------------------------------------------------------
require 'pl'
require 'nn'
require 'cudnn'
require 'cunn'
local opts = require 'opts'
local DataLoader = require 'data/dataloader'

torch.setdefaulttensortype('torch.FloatTensor')
----------------------------------------------------------------------
-- Get the input arguments parsed and stored in opt
opt = opts.parse(arg)
print(opt)

--print(cutorch.getDeviceProperties(opt.devid))
cutorch.setDevice(opt.devid)
print("Folder created at " .. opt.save)
os.execute('mkdir -p ' .. opt.save)

----------------------------------------------------------------------
print '==> load modules'
local data, chunks, ft

-- data loading
local trainLoader, valLoader = DataLoader.create(opt)
opt.classes = trainLoader.classes

-- save opt to file
print 'saving opt as txt and t7'
local filename = paths.concat(opt.save,'opt.txt')
local file = io.open(filename, 'w')
for i,v in pairs(opt) do
    file:write(tostring(i)..' : '..tostring(v)..'\n')
end
file:close()
torch.save(path.join(opt.save,'opt.t7'),opt)

----------------------------------------------------------------------
print '==> training!'
local epoch = 1

-- create model
t = paths.dofile("models/"..opt.model..".lua")

local train = require 'train'
local test  = require 'test'
local besterr = 9999999999
while epoch < opt.maxepoch do
    print("----- epoch # " .. epoch)
   local trainConf, model, loss = train(trainLoader, epoch)
   besterr = test(valLoader, epoch, trainConf, model, loss, besterr )
   -- trainConf = nil
   collectgarbage()
   epoch = epoch + 1
end
