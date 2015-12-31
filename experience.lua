local _ = require 'moses'

local experience = {}

-- Creates experience replay memory
experience.create = function(opt)
  local memory = {}
  local stateSize = torch.LongStorage({opt.memSize, opt.histLen, opt.nChannels, opt.height, opt.width}) -- Calculate state storage size
  -- Allocate memory for experience
  memory.states = torch.FloatTensor(stateSize) -- ByteTensor uses less memory but reduces speed from byte <-> float conversion needed
  memory.actions = torch.ByteTensor(opt.memSize) -- Discrete action indices
  memory.rewards = torch.FloatTensor(opt.memSize) -- Stored at time t
  -- Terminal conditions stored at time t+1, encoded by 0 = false, 1 = true
  memory.terminals = torch.ByteTensor(opt.memSize):fill(1) -- Filling with 1 prevents going back in history initially
  -- Internal pointer
  memory.index = 1
  memory.isFull = false
  -- TD-error δ-based priorities
  memory.priorities = torch.FloatTensor(opt.memSize):fill(0) -- Stored at time t
  local smallConst = 1e-6 -- Account for half precision
  memory.maxPriority = opt.tdClip -- Should prioritise sampling experience that has not been learnt from

  -- Initialise first time step
  memory.states[1]:zero() -- Blank state
  memory.actions[1] = 1 -- Action is no-op

  -- Calculates circular indices
  local circIndex = function(x)
    local ind = x % opt.memSize
    return ind == 0 and opt.memSize or ind -- Correct 0-index
  end

  -- Returns number of saved tuples
  function memory:size()
    return self.isFull and opt.memSize or self.index - 1
  end

  -- Stores experience tuple parts (including pre-emptive action)
  function memory:store(reward, state, terminal, action)
    self.rewards[self.index] = reward
    -- Store with maximal priority
    self.priorities[self.index] = self.maxPriority + smallConst
    self.maxPriority = self.maxPriority + smallConst

    -- Increment index
    self.index = self.index + 1
    -- Circle back to beginning if memory limit reached
    if self.index > opt.memSize then
      self.isFull = true -- Full memory flag
      self.index = 1 -- Reset index
    end

    self.states[self.index] = state:float()
    self.terminals[self.index] = terminal and 1 or 0
    self.actions[self.index] = action
  end

  -- Converts a CDF from a PDF
  local pdfToCdf = function(pdf)
    local c = 0
    pdf:apply(function(x)
      c = c + x
      return c
    end)
  end

  -- Returns indices and importance-sampling weights based on (stochastic) proportional prioritised sampling
  function memory:sample(nSamples, priorityType)
    local N = self:size()
    local indices, w

    -- Priority 'none' = uniform sampling
    if priorityType == 'none' then
      indices = torch.randperm(N):long()
      indices = indices[{{1, nSamples}}]
      w = torch.ones(nSamples) -- Set weights to 1 as no correction needed
    else
      -- Calculate sampling probability distribution P
      local P = torch.pow(self.priorities[{{1, N}}], opt.alpha) -- Use prioritised experience replay exponent α
      local Z = torch.sum(P) -- Calculate normalisation constant
      P:div(Z) -- Normalise

      -- Calculate importance-sampling weights w
      w = torch.mul(P, N):pow(-opt.beta[opt.step]) -- Use importance-sampling exponent β
      w:div(torch.max(w)) -- Normalise weights so updates only scale downwards (for stability)

      -- Create a cumulative distribution for inverse transform sampling
      pdfToCdf(P) -- Convert distribution
      indices = torch.sort(torch.Tensor(nSamples):uniform()) -- Generate uniform numbers for sampling
      -- Perform linear search to sample
      local minIndex = 1
      for i = 1, nSamples do
        while indices[i] > P[minIndex] do
          minIndex = minIndex + 1
        end
        indices[i] = minIndex -- Get sampled index
      end
      indices = indices:long() -- Convert to LongTensor for indexing
      w = w:index(1, indices) -- Index weights
    end

    if opt.gpu > 0 then
      w = w:cuda()
    end

    return indices, w
  end

  -- Retrieves experience tuples (s, a, r, s', t)
  function memory:retrieve(tuple, indices)
    local batchSize
    if not indices then
      indices = tuple
      batchSize = indices:size(1)
      tuple = {
        states = opt.Tensor(batchSize, opt.histLen, opt.nChannels, opt.height, opt.width),
        actions = torch.ByteTensor(batchSize),
        rewards = opt.Tensor(batchSize),
        transitions = opt.Tensor(batchSize, opt.histLen, opt.nChannels, opt.height, opt.width),
        terminals = torch.ByteTensor(batchSize)
      }
    else
      batchSize = indices:size(1)
    end

    for i = 1, batchSize do
      -- Retrieve state history
      tuple.states[i] = self.states[indices[i]] -- Assume indices are valid
      -- Retrieve action
      tuple.actions[i] = self.actions[indices[i]]
      -- Retrieve rewards
      tuple.rewards[i] = self.rewards[indices[i]]
      -- Retrieve terminal status
      tuple.terminals[i] = self.terminals[indices[i]]

      -- If not terminal, fill in transition history
      if tuple.terminals[i] == 0 then
        tuple.transitions[i] = self.states[circIndex(indices[i] + 1)]
      end
    end

    return tuple
  end

  -- Update experience priorities using TD-errors δ
  function memory:updatePriorities(indices, delta)
    local priorities = delta:float()
    if opt.memPriority == 'proportional' then
      priorities:abs()
    end

    for p = 1, indices:size(1) do
      self.priorities[indices[p]] = priorities[p] + smallConst -- Allows transitions to be sampled even if error is 0
    end
  end

  return memory
end

return experience
