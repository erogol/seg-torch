--
-- Read the given dataset path and create a data structure
-- Data link: https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid
--

require 'xlua'
require 'image'
local sys = require 'sys'
local ffi = require 'ffi'

local M = {}

local function findImages(dir)
    local imagePath = torch.CharTensor()
    local imageClass = torch.LongTensor()

    ----------------------------------------------------------------------
    -- Options for the GNU and BSD find command
    local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP', 'tif'}
    local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
    for i=2,#extensionList do
        findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
    end

    -- Find all the images using the find command
    local f = io.popen('find -L ' .. dir .. findOptions .. '| sort')

    local maxLength = -1
    local imagePaths = {}
    local imageClasses = {}
    local counter = 0

    -- Generate a list of all the images and their class
    while true do
        local line = f:read('*line')
        if not line then break end
        counter = counter + 1
        if counter % 100 == 0 then print(counter) end

        table.insert(imagePaths, line)

        maxLength = math.max(maxLength, #line + 1)
    end
    print(imagePaths)

    f:close()

    -- Convert the generated list to a tensor for faster loading
    local nImages = #imagePaths
    local imagePath = torch.CharTensor(nImages, maxLength):zero()
    for i, path in ipairs(imagePaths) do
        ffi.copy(imagePath[i]:data(), path)
    end

    return imagePath, imagePaths
end

local function setDatasetStats(imagePaths, maskPaths, classes)

    -- compute image mean and std
    print(' | Estimating mean, std, class weights...')
    local meanEstimate = {0,0,0}
    local stdEstimate = {0,0,0}
    local classWeights = torch.Tensor(#classes):fill(0)
    for i=1,#maskPaths do
        -- compute mean std
        local imagePath = imagePaths[i]
        local img = image.load(imagePath, 3, 'float')
        for j=1,3 do
            meanEstimate[j] = meanEstimate[j] + img[j]:mean()
            stdEstimate[j] = stdEstimate[j] + img[j]:std()
        end

        -- compute class weights
        local maskPath = maskPaths[i]
        local mask = image.load(maskPath, 1, 'byte'):squeeze():float() + 2
        local tmp = mask:eq(13):float()
        mask = mask - tmp * #classes
        classWeights = classWeights + torch.histc(mask, #classes, 1, #classes)
        xlua.progress(i, #maskPaths)
    end
    for j=1,3 do
        meanEstimate[j] = meanEstimate[j] / #maskPaths
        stdEstimate[j] = stdEstimate[j] / #maskPaths
    end
    local mean = meanEstimate
    local std = stdEstimate
    local meanstd = {["mean"]=mean, ["std"]=std}
    print(' | mean, std:')
    print(meanstd)

    -- norm class weights
    local normHist = classWeights / classWeights:median()[1]
    local normClassWeights = torch.Tensor(#classes):fill(1)
    for i = 1, #classes do
        if classWeights[i] < 1 or i == 1 then -- ignore unlabeled
            normClassWeights[i] = 0
        else
            normClassWeights[i] = 1 / (torch.log(1.2 + normHist[i]))
        end
    end
    print(' | class weights:')
    print(normClassWeights)

    collectgarbage()
    return mean, std, normClassWeights
end

function M.exec(opt, cacheFile)
    -- define class maps
    local classes = {'Void', 'Sky', 'Building', 'Column-Pole',
    'Road', 'Sidewalk', 'Tree', 'Sign-Symbol',
    'Fence', 'Car', 'Pedestrian', 'Bicyclist'}

    -- find the image path names
    print(" | finding all images")
    local trainImagePaths, trainImagePathsList = findImages(opt.datapath .. '/train')
    local trainMaskPaths, trainMaskPathsList = findImages(opt.datapath .. '/trainannot')
    local valImagePaths, tmp1 = findImages(opt.datapath .. '/val')
    local valMaskPaths, tmp2 = findImages(opt.datapath .. '/valannot')

    -- set class weights
    local mean, std, classWeights = setDatasetStats(trainImagePathsList, trainMaskPathsList, classes)

    -- sanity check
    assert(trainImagePaths:size(1) == trainMaskPaths:size(1))
    assert(valMaskPaths:size(1) == valImagePaths:size(1))

    if trainImagePaths:nElement() == 0 then
        error(" !!! Check datapath, it might be wrong !!! ")
    end

    local info = {
        basedir = opt.dataPath,
        classes = classes,
        classWeights = classWeights,
        mean = mean,
        std = std,
        train = {
            imagePath = trainImagePaths,
            labelPath = trainMaskPaths,
        },
        val = {
            imagePath = valImagePaths,
            labelPath = valMaskPaths,
        },
    }

    print(" | saving list of images to " .. cacheFile)
    print(" | number of traning images: ".. trainImagePaths:size(1))
    print(" | number of validation images: ".. valImagePaths:size(1))
    torch.save(cacheFile, info)
    return info
end

return M
