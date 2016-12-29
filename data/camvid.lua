local image = require 'image'
local paths = require 'paths'
local t = require 'data/transforms'
local ffi = require 'ffi'

local M = {}
local CamvidDataset = torch.class('segnet.CamvidDataset', M)

function CamvidDataset:__init(imageInfo, opt, split)
    self.imageInfo = imageInfo[split]
    self.classes = imageInfo.classes
    self.opt = opt
    self.split = split
    collectgarbage()
end

function CamvidDataset:get(i)
    -- return a new input, target couple

    local image_path = ffi.string(self.imageInfo.imagePath[i]:data())
    local label_path = ffi.string(self.imageInfo.labelPath[i]:data())

    local image = self:_loadImage(image_path, 3)
    local label = self:_loadImage(label_path, 1):squeeze():float() + 2
    local mask = label:eq(13):float()
    label = label - mask * #self.classes

    return {
        input = image,
        target = label,
        input_path = image_path,
        target_path = label_path
    }
end

function CamvidDataset:_loadImage(path, channels)
    local ok, input = pcall(function()
        if channels == 1 then
            return image.load(path, channels, 'byte')
        else
            return image.load(path, channels, 'float')
        end
    end)

    -- Sometimes image.load fails because the file extension does not match the
    -- image format. In that case, use image.decompress on a ByteTensor.
    if not ok then
        local f = io.open(path, 'r')
        assert(f, 'Error reading: ' .. tostring(path))
        local data = f:read('*a')
        f:close()

        local b = torch.ByteTensor(string.len(data))
        ffi.copy(b:data(), data, b:size(1))

        input = image.decompress(b, channels, 'float')
    end

    return input
end

function CamvidDataset:size()
    return self.imageInfo.imagePath:size(1)
end

function CamvidDataset:preprocess()
    if self.split == 'train' then
        return t.Compose{
            t.ColorNormalize(self.opt.mean, self.opt.std),
            t.Scale(360),
            t.HorizontalFlip(0.5)
        }
    elseif self.split == 'val' then
        return t.Compose{
            t.ColorNormalize(self.opt.mean, self.opt.std),
            t.Scale(360)
            --   t.RandomCrop(self.opt.imHeight, self.opt.imWidth)
        }
    else
        error('invalid split: ' .. self.split)
    end
end

return M.CamvidDataset
