import random, torch
import torch.nn as nn

# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(3, 3)
        self.fc2 = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x

# Define the agent
class Agent:
    def __init__(self):
        self.neural_network = NeuralNetwork()
        self.fitness = 0

# Genetic Algorithm class
class GeneticAlgorithm:
    def __init__(
            self, 
            pop_size: int, 
            generations: int, 
            threshold: float, 
            input_data: torch.tensor, 
            target: torch.tensor
    ):
        self.pop_size = pop_size
        self.generations = generations
        self.threshold = threshold
        self.input_data = input_data
        self.target = target

    # Generates the population
    def generate_agents(self) -> None:
        self.agents = [Agent() for _ in range(self.pop_size)]

    # Calculate the fitness score with MSELoss() functin from PyTorch
    def fitness(self) -> None:
        for agent in self.agents:
            output = agent.neural_network.forward(self.input_data)
            loss = nn.MSELoss()(output, self.target)
            agent.fitness = loss.item()

    # Survival of the fittest
    def selection(self) -> None:
        agents = sorted(self.agents, key=lambda agent: agent.fitness, reverse=False)
        agents = agents[:int(0.2 * len(agents))]
        self.agents = agents
        
    # Choose random parents
    def choose_parents(self) -> list[Agent]:
        return [random.choice(self.agents), random.choice(self.agents)]
    
    # Generate new children
    def generate_children(self) -> list[Agent]:
        return [Agent() for _ in range(2)]

    # Assign genes to the new agents from their parents
    def crossover(self) -> None:
        offspring = []
        
        for _ in range((self.pop_size - len(self.agents)) // 2):
            parent1, parent2 = self.choose_parents()
            child1, child2 = self.generate_children()

            for parent1_param, parent2_param, child1_param, child2_param in zip(parent1.neural_network.parameters(),
                                                                                parent2.neural_network.parameters(),
                                                                                child1.neural_network.parameters(),
                                                                                child2.neural_network.parameters()):
                split = random.randint(0, (len(parent1_param.data) - 1))
                child1_param.data[:split].copy_(parent1_param.data[:split])
                child1_param.data[split:].copy_(parent2_param.data[split:])
                child2_param.data[:split].copy_(parent2_param.data[:split])
                child2_param.data[split:].copy_(parent1_param.data[split:])

            offspring.extend([child1, child2])
        
        self.agents.extend(offspring)

    # Mutate the gene pool
    def mutation(self) -> None:
        for agent in self.agents:
            if random.uniform(0.0, 1.0) <= self.threshold:
                for param in agent.neural_network.parameters():
                    noise = torch.randn(param.data.size())
                    param.data += noise

    def train(self) -> Agent:
        self.generate_agents()
        threshold_met = False
        
        for i in range(self.generations):
            self.fitness()
            self.selection()
            self.crossover()
            self.mutation()
            self.fitness()

            if not threshold_met:
                if any(agent.fitness < self.threshold for agent in self.agents):
                    print('Threshold met at generation '+str(i)+'!')
                    threshold_met = True
            
            if i % 10==0:
                print('Generation',str(i),':')
                print('The Best agent has fitness ' +str(self.agents[0].fitness)+ 'at generation '+str(i)+'.')
                print('The Worst agent has fitness ' +str(self.agents[-1].fitness)+ 'at generation '+str(i)+'.')

        return self.agents[0]

if __name__ == "__main__":
    input_data = torch.tensor(
        [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]],
        dtype=torch.float32
    )
    target = torch.tensor(
        [[0, 1, 1, 0]],
        dtype=torch.float32
    ).T


    ga = GeneticAlgorithm(
        pop_size=100,
        generations=50,
        threshold=0.1,
        input_data=input_data,
        target=target
    )
    agent = ga.train()
    print(agent.fitness)
    print(agent.neural_network.forward(input_data))