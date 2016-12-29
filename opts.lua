local opts = {}

lapp = require 'pl.lapp'
function opts.parse(arg)
   local opt = lapp [[
   Command line options:
   Training Related:
   -l,--learningRate       (default 5e-4)        learning rate
   -d,--learningRateDecaySteps  (default 5)   number of epochs to reduce LR by 0.1
   -w,--weightDecay        (default 2e-4)        L2 penalty on the weights
   -m,--momentum           (default 0.9)         momentum
   -b,--batchSize          (default 10)          batch size
   --maxepoch              (default 300)         maximum number of training epochs
   --plot                                        plot training/testing error in real-time
   --manualSeed            (default 0),          Manually set RNG seed
   --checkpoint            (default false)       Save model per epoch

   Device Related:
   -t,--nThreads           (default 8)           number of threads for data loader
   -i,--devid              (default 1)           device ID (if using CUDA)
   --nGPU                  (default 1)           number of GPUs you want to train on
   --save                  (default /media/)     save trained model here

   Dataset Related:
   --channels              (default 3)
   --datapath              (default /media/Dataset/)
                           dataset location
   --dataset               (default nil)         dataset type
   --cachepath             (default /media/)     cache directory to save the loaded dataset
   --imHeight              (default 360)         image height
   --imWidth               (default 480)         image width
   --maskHeight            (default 45)          mask height
   --maskWidth             (default 60)          mask width

   Model Related:
   --model                 (default models/encoder.lua)
                           Path of a model
   --encoder               (default /media/Models/CamVid/enc/model-100.net)
                           pretrained encoder for which you want to train your decoder

   Saving/Displaying Information:
   --confusion             (default skip)
                           skip: skip confusion, all: test+train, test : test only
 ]]

   return opt
end

return opts
