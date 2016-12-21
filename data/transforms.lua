--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Image transforms for data augmentation and input normalization
--

require 'image'

local M = {}

function M.Compose(transforms)
    return function(input, label)
        for _, transform in ipairs(transforms) do
            input, label = transform(input, label)
        end
        return input, label
    end
end

function M.ColorNormalize(mean, std)
    return function(input, label)
        input = input:clone()
        label = label:clone()
        for i=1,3 do
            input[i]:add(-mean[i])
            input[i]:div(std[i])
        end
        return input, label
    end
end

-- Scales the smaller edge to size
function M.Scale(size, interpolation)
    interpolation = interpolation or 'bicubic'
    return function(input, label)
        local w, h = input:size(3), input:size(2)
        if (w <= h and w == size) or (h <= w and h == size) then
            return input, label
        end
        if w < h then
            return image.scale(input, size, h/w * size, interpolation), image.scale(label, size, h/w * size, 'simple')
        else
            return image.scale(input, w/h * size, size, interpolation), image.scale(label, w/h * size, size, 'simple')
        end
    end
end

-- Crop to centered rectangle
function M.CenterCrop(target_h, target_w)
    return function(input, label)
        local w1 = math.ceil((input:size(3) - target_w)/2)
        local h1 = math.ceil((input:size(2) - target_h)/2)
        return image.crop(input, w1, h1, w1 + target_w, h1 + target_h), image.crop(label, w1, h1, w1 + target_w, h1 + target_h) -- center patch
    end
end

function sleep(n)
    os.execute("sleep " .. tonumber(n))
end

-- Random crop form larger image with optional zero padding
function M.RandomCrop(sizeH, sizeW)
    padding = padding or 0

    return function(input, label)

        function vector_unique(input_table)
            local unique_elements = {} --tracking down all unique elements
            local output_table = {} --result table/vector

            for _, value in ipairs(input_table) do
                unique_elements[value] = true
            end

            for key, _ in pairs(unique_elements) do
                table.insert(output_table, key)
            end

            return output_table
        end

        local cont_flag = true
        local input_img = nil
        local target_img = nil

        -- continue cropping unless found non-zero pixels in the crop
        -- this prevents crops with only background
        while cont_flag do
            local w, h = input:size(3), input:size(2)
            if w == size and h == size then
                return input, label
            end

            local x1, y1 = torch.random(0, w - sizeW), torch.random(0, h - sizeH)
            input_img = image.crop(input, x1, y1, x1 + sizeW, y1 + sizeH)
            assert(input_img:size(2) == sizeW and input_img:size(3) == sizeH, 'wrong crop size')

            target_img = image.crop(label, x1, y1, x1 + sizeW, y1 + sizeH)
            assert(target_img:size(2) == sizeW and target_img:size(3) == sizeH, 'wrong crop size')

            if target_img:sum() > 0 then
                cont_flag = false
            end
        end

        -- image.d("deneme_input.jpg",input_img)
        -- local uniques = vector_unique(target_img:storage():totable())
        -- for k, v in pairs( uniques ) do
        --     print(k, v)
        -- end
        -- print(target_img:sum())
        return input_img, target_img
    end
end

function M.HorizontalFlip(prob)
    return function(input, label)
        if torch.uniform() < prob then
            input = image.hflip(input)
            label = image.hflip(label)
        end
        return input, label
    end
end

function M.SetLabelSize(maskWidth, maskHeight)
    return function(input, label)
        label = image.scale(label, maskWidth, maskHeight, "simple")
        return input, label
    end
end

function M.Rotation(deg)
    return function(input, label)
        if deg ~= 0 then
            angle = (torch.uniform() - 0.5) * deg * math.pi / 180
            input = image.rotate(input, angle, 'bilinear')
            label = image.rotate(label, angle, 'simple')
        end
        return input, label
    end
end

local function grayscale(dst, img)
    dst:resizeAs(img)
    dst[1]:zero()
    dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
    dst[2]:copy(dst[1])
    dst[3]:copy(dst[1])
    return dst
end

function M.OneHotLabel(nclasses)
    return function(input, label)
        local oneHot = torch.ByteTensor(nclasses,label:size(2),label:size(3))
        for i = 1,nclasses do
            oneHot[i] = label:eq(i):byte()
        end
        label = oneHot:clone()
        return input, label
    end
end

function M.CatLabel()
    return function(input, label)
        _, ar = torch.max(label, 1)
        label = ar[1]
        return input, label
    end
end

--- [[Structural Noise]] ---
function M.ElasticTransform(alpha, sigma)
    return function (input, label)
        H = input:size(2)
        W = input:size(3)
        filterSize = math.max(5,math.ceil(3*sigma))

        flow = torch.rand(2, H, W)*2 - 1
        kernel = image.gaussian(filterSize, sigma, 1, true)
        flow = image.convolve(flow, kernel, 'same')*alpha

        return image.warp(input, flow), image.warp(label, flow)
    end
end

return M
