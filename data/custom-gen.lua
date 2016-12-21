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

    return imagePath
end

function M.exec(opt, cacheFile)
    -- define class maps
    local classes = {}
    classes[1] = "Unlabeled" -- idx 0 is ignored
    classes[2] = "Building"

    -- find the image path names
    print(" | finding all images")
    local ImagePath = findImages(opt.datapath .. '/images')
    local LabelPath = findImages(opt.datapath .. '/masks')

    -- dummy split since labels don't matter

    if ImagePath:nElement() == 0 then
        print("no images found in the directory, probably wrong path")
        N = 0
    else
        N = ImagePath:size(1)
    end

    local info = {
        basedir = opt.dataPath,
        classes = classes,
        train = {
            imagePath = ImagePath[{{1,N*0.8},{}}],
            labelPath = LabelPath[{{1,N*0.8},{}}],
        },
        val = {
            imagePath = ImagePath[{{N*0.8+1,N},{}}],
            labelPath = LabelPath[{{N*0.8+1,N},{}}],
        },
    }

    print(" | saving list of images to " .. cacheFile)
    print(" | number of traning images: ".. ImagePath[{{1,N*0.8},{}}]:size(1))
    print(" | number of validation images: ".. ImagePath[{{N*0.8+1,N},{}}]:size(1))
    torch.save(cacheFile, info)
    return info
end

return M
