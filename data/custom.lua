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
   self.dir = opt.datapath
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
   collectgarbage()
end

function CamvidDataset:get(i)

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

   local image_path = ffi.string(self.imageInfo.imagePath[i]:data())
   local label_path = ffi.string(self.imageInfo.labelPath[i]:data())

   local image = self:_loadImage(paths.concat(self.dir, image_path), 3)
   local label = self:_loadImage(paths.concat(self.dir, label_path), 1)/254
   label = label + 1

   unique_labels = vector_unique(label:storage():totable())
   num_labels = #unique_labels

   -- check for nay possible error
   assert(num_labels <= #self.classes)

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

-- -- Computed from random subset of Camvid training images
-- local meanstd = {
--    mean = { 0.485, 0.456, 0.406 },
--    std = { 0.229, 0.224, 0.225 },
-- }
-- local pca = {
--    eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
--    eigvec = torch.Tensor{
--       { -0.5675,  0.7192,  0.4009 },
--       { -0.5808, -0.0045, -0.8140 },
--       { -0.5836, -0.6948,  0.4203 },
--    },
-- }

function CamvidDataset:preprocess()
   if self.split == 'train' then
      return t.Compose{
           t.Scale(270),
        --    t.RandomCrop(self.opt.imHeight, self.opt.imWidth),
        --    t.SetLabelSize(self.opt.maskWidth, self.opt.maskHeight),
        --    t.OneHotLabel(#self.classes),
           t.HorizontalFlip(0.5),
      }
   elseif self.split == 'val' then
      return t.Compose{
          t.Scale(270)
        --   t.RandomCrop(self.opt.imHeight, self.opt.imWidth)
      }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.CamvidDataset
