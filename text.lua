function sets_cover(actionSets)
  local num_observation = actionSets:size()[1]
  local num_agents = actionSets:size()[2]
  local num_actions = actionSets:size()[3]
  local submodular_agents = torch.Tensor(num_agents):fill(0)
  local uncovered = torch.Tensor(num_observation, num_agents, num_agents, num_actions):fill(1)
  local scores = torch.Tensor(1):fill(-1)
  local best_agent = 1
  while not (scores:max() == 0) do
    local agent_cs = torch.reshape(actionSets, num_observation, num_agents, 1, num_actions):repeatTensor(1, 1, num_agents, 1)
    local aggregate_cs = torch.reshape(actionSets, num_observation, 1, num_agents, num_actions):repeatTensor(1, num_agents, 1, 1)
    local available_cs = torch.cmul(agent_cs, aggregate_cs):cmul(uncovered)
    scores = available_cs:max(4):sum(1):sum(3):reshape(num_agents)
    _, best_agent = scores:max(1)
    if not (scores:max() == 0) then
      submodular_agents[best_agent[1]] = 1
      uncovered = torch.cmul(uncovered, (1 - agent_cs:narrow(2, best_agent[1],1):repeatTensor(1,num_agents,1,1)))
    end
  end 
  return submodular_agents
end

x = torch.Tensor(5,5,4):random(0,1)

print("actionSets:", x)

y = sets_cover(x)

print("sets_cover:", y)