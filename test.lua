require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

torch.setdefaulttensortype('torch.FloatTensor')

errorLogger = optim.Logger(paths.concat(opt.save, 'error.log'))
coTotalLogger = optim.Logger(paths.concat(opt.save, 'confusionTotal.log'))
coAveraLogger = optim.Logger(paths.concat(opt.save, 'confusionAvera.log'))
coUnionLogger = optim.Logger(paths.concat(opt.save, 'confusionUnion.log'))

print '==> Creating Test function...'
local teconfusion, filename
teconfusion = optim.ConfusionMatrix(opt.classes)

print ' | ==> allocating minibatch memory'
local x
local yt

function test(dataloader, epoch, trainConf, model, loss, besterr )
    local classes = opt.classes
    local dataSize = dataloader:size()
    local err = 0
    local totalerr = 0

    local function copyInputs(sample)
        x =  x or torch.CudaTensor()
        yt = yt or torch.CudaTensor()

        x:resize(sample.input:size()):copy(sample.input)
        yt:resize(sample.target:size()):copy(sample.target)
    end
    model:evaluate()
    print('==> Testing:')
    for n, sample in dataloader:run() do
        xlua.progress(n, dataSize)
        copyInputs(sample)
        local y = model:forward(x)
        err = loss:forward(y,yt)
        if opt.noConfusion == 'test' or opt.noConfusion == 'all' then
            local y = y:transpose(2, 4):transpose(2, 3)
            y = y:reshape(y:numel()/y:size(4), #classes)
            local _, predictions = y:max(2)
            predictions = predictions:view(-1)
            local k = yt:view(-1)
            teconfusion:batchAdd(predictions, k)
        end
        totalerr = totalerr + err
        collectgarbage()
    end
    totalerr = totalerr / (dataSize / opt.batchSize)
    print('Test Error: ', totalerr )
    errorLogger:add{['Training error'] = trainError,
    ['Testing error'] = totalerr}
    if opt.plot then
        errorLogger:style{['Training error'] = '-',
        ['Testing error'] = '-'}
        errorLogger:plot()
    end
    if totalerr < besterr then
        filename = paths.concat(opt.save, 'model-best.net')
        print('==> saving BEST model to '..filename)

        torch.save(filename, model:clearState())
        -- update to min error:
        if opt.noConfusion == 'test' or opt.noConfusion == 'all' then
            coTotalLogger:add{['confusion total accuracy'] = teconfusion.totalValid * 100 }
            coAveraLogger:add{['confusion average accuracy'] = teconfusion.averageValid * 100 }
            coUnionLogger:add{['confusion union accuracy'] = teconfusion.averageValid * 100 }

            filename = paths.concat(opt.save,'confusion-'..epoch..'.t7')
            print('==> saving confusion to '..filename)
            torch.save(filename,teconfusion)

            filename = paths.concat(opt.save, 'confusionMatrix-best.txt')
            print('==> saving confusion matrix to ' .. filename)
            local file = io.open(filename, 'w')
            file:write("--------------------------------------------------------------------------------\n")
            file:write("Training:\n")
            file:write("================================================================================\n")
            file:write(tostring(trainConf))
            file:write("\n--------------------------------------------------------------------------------\n")
            file:write("Testing:\n")
            file:write("================================================================================\n")
            file:write(tostring(teconfusion))
            file:write("\n--------------------------------------------------------------------------------")
            file:close()
        end
        filename = paths.concat(opt.save, 'best-number.txt')
        local file = io.open(filename, 'w')
        file:write("----------------------------------------\n")
        file:write("Best test error: ")
        file:write(tostring(totalerr))
        file:write(", in epoch: ")
        file:write(tostring(epoch))
        file:write("\n----------------------------------------\n")
        file:close()
        besterr = totalerr
    end
    -- cudnn.convert(model, nn)
    local filename = paths.concat(opt.save, 'model-'..epoch..'.net')
    if opt.checkpoint then
        print('==> saving model checkpoint to '..filename)
        torch.save(filename, model:clearState())
    end
    if opt.noConfusion == 'test' or opt.noConfusion == 'all' then
        -- update to min error:
        filename = paths.concat(opt.save, 'confusionMatrix-' .. epoch .. '.txt')
        print('==> saving confusion matrix to ' .. filename)
        local file = io.open(filename, 'w')
        file:write("--------------------------------------------------------------------------------\n")
        file:write("Training:\n")
        file:write("================================================================================\n")
        file:write(tostring(trainConf))
        file:write("\n--------------------------------------------------------------------------------\n")
        file:write("Testing:\n")
        file:write("================================================================================\n")
        file:write(tostring(teconfusion))
        file:write("\n--------------------------------------------------------------------------------")
        file:close()
        filename = paths.concat(opt.save, 'confusion-'..epoch..'.t7')
        print('==> saving test confusion object to '..filename)
        torch.save(filename,teconfusion)
        trainConf:zero()
        teconfusion:zero()
    end
    collectgarbage()
    return besterr
end

-- Export:
return test
