-- Require the 'nn' function for neural network functions
require 'nn';
nExamples = 10000

trainset = {}
trainset.data = torch.Tensor(nExamples,64,1):zero() -- Data will be sized as 5000x64x1
trainset.label = torch.Tensor(nExamples):zero()     -- Use one dimensional tensor for label

--The network trainer expects an index metatable
setmetatable(trainset,
{__index = function(t, i)
    return {t.data[i], t.label[i]}  -- The trainer is expecting trainset[123] to be {data[123], label[123]}
    end}
);

--The network trainer expects a size function
function trainset:size()
    return self.data:size(1)
end

function GenerateTrainingSet()

    -- Time to prepare the training set with data
    -- At random, have data be either a triangular pulse, or a rectangular pulse
    -- Have randomness as to when the signal starts, ends, and how high it is
    for i=1,nExamples do
        curWaveType = math.random(1,2)      -- 1 for triangular signal, 2 for square pulse
        curWaveHeight = math.random(5,10)   -- how high is signal
        curWaveWidth = math.random(20,40)   -- how wide is signal
        curWaveStart = math.random(5,10)    -- when to start signal

        for j=1,curWaveStart-1 do
            trainset.data[i][j][1] = 0
        end

        if curWaveType==1 then   -- We are making a triangular wave
            delta = curWaveHeight / (curWaveWidth/2);
            for curIndex=1,curWaveWidth/2 do
                trainset.data[i][curWaveStart-1+curIndex][1] = delta * curIndex
            end
            for curIndex=(curWaveWidth/2)+1, curWaveWidth do
                trainset.data[i][curWaveStart-1+curIndex][1] = delta * (curWaveWidth-curIndex)
            end
            trainset.label[i] = 1
        else
            for j=1,curWaveWidth do
                trainset.data[i][curWaveStart-1+j][1] = curWaveHeight
            end
            trainset.label[i] = 2
        end
    end
end

GenerateTrainingSet()

-- This is where we build the model
model = nn.Sequential()                       -- Create network

-- First convolution, using ten, 7-element kernels
model:add(nn.TemporalConvolution(1, 10, 7))   -- 64x1 goes in, 58x10 goes out
model:add(nn.TemporalMaxPooling(2))           -- 58x10 goes in, 29x10 goes out
model:add(nn.ReLU())                          -- non-linear activation function

-- Second convolution, using 5, 7-element kernels
model:add(nn.TemporalConvolution(10, 5, 7))   -- 29x10 goes in, 23x5 goes out
model:add(nn.TemporalMaxPooling(2))           -- 23x5 goes in, 11x5 goes out
model:add(nn.ReLU())                          -- non-linear activation function

-- After convolutional layers, time to do fully connected network
model:add(nn.View(11*5))                        -- Reshape network into 1D tensor

model:add(nn.Linear(11*5, 30))                  -- Fully connected layer, 55 inputs, 30 outputs
model:add(nn.ReLU())                            -- non-linear activation function

model:add(nn.Linear(30, 2))                     -- Final layer has 2 outputs. One for triangle wave, one for square
model:add(nn.ReLU())                            -- non-linear activation function
model:add(nn.LogSoftMax())                      -- log-probability output, since this is a classification problem

print(#model:forward(torch.Tensor(64,1)))

criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.01
trainer.maxIteration = 200 -- do 200 epochs of training

trainer:train(trainset)
