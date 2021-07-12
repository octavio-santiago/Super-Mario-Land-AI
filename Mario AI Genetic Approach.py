import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import pandas as pd
import random
import os
import sys
from pyboy import PyBoy, WindowEvent
import json

style.use("ggplot")

classes = 10
batch_size = 64
population = 5
generations = 5
threshold = 100000


games = 800
time_h = 0
lucro0 = 0

class environment:
    def __init__(self):
        #start the game
        filename = 'roms/Super Mario Land.gb'
        quiet = "--quiet" in sys.argv
        self.pyboy = PyBoy(filename, window_type="headless" if quiet else "SDL2", window_scale=3, debug=quiet, game_wrapper=True)
        self.pyboy.set_emulation_speed(1)
        assert self.pyboy.cartridge_title() == "SUPER MARIOLAN"

        self.mario = self.pyboy.game_wrapper()
        self.mario.start_game()
        self.done= False
        self.time = 10

        assert self.mario.score == 0
        assert self.mario.lives_left == 2
        assert self.mario.time_left == 400
        assert self.mario.world == (1, 1) #stage
        assert self.mario.fitness == 0 # A built-in fitness score for AI development
        
        #set state and action size
        self.action_size = 5 #number of possible actions
        state_full = np.asarray(self.mario.game_area())
        np.append(state_full,self.mario.level_progress)
        self.state_size = state_full.size

        #print(self.mario)        
        
    def reset(self):
        self.mario.reset_game() #back to the last state saved
        self.done = False
        self.pyboy.tick()
        #assert self.mario.fitness == 0 # A built-in fitness score for AI development
        assert self.mario.lives_left == 2
        self.position = self.mario.level_progress
        state_full = np.asarray(self.mario.game_area())
        np.append(state_full,self.mario.level_progress)
        
        return state_full

    def step(self,action):
        if action == 0:
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self.time = 50
        elif action == 1:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
            self.time = 5
        elif action == 2:
            #self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            #time.sleep(0.1)
            #self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            self.time = 5
        elif action == 3:
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
            self.time = 5
        elif action == 4:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
            self.time = 5
                 
        return action, self.time



class Network():
    def __init__(self):
        self.actions = []
        self.generation = 0
        #generate random actions
        for i in range(games):
            #self.action = env.action_space.sample()
            self.action = random.randint(0,5)
            self.actions.append(self.action)

        self.lucro = 0

    def get_actions(self):
        return self.actions

    def set_actions(self,actions,lucro):
        self.actions = actions
        self.lucro = lucro
        return self.lucro

def init_networks(population):
    return [Network() for _ in range(population)]


def fitness(networks):
    for network in networks:
        #init env
        env = environment()
        state_size = env.state_size
        action_size = env.action_size
        fitness = env.mario.fitness
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        
        actions = network.get_actions()
        strategies = []
        lucro_tot = 0
        
        #try:
        time_h = 0
        # loop through actions
        for act in actions:
            #print("action: ", act, " lives: ",env.mario.lives_left, " pos: ", env.mario.level_progress)
            try:
                #16,17,18,19,27,26
                filteredMario = [x for x in list(state[0]) if (x>10 and x<30)]
                #print(filteredMario)
                index_mario = list(state[0]).index(filteredMario[0])
                feet_val = state[0][index_mario + 20]
                #print('POSITION: ',state[0][index_mario], state[0][index_mario + 20])
            except:
                break

            act, tempo = env.step(act)
            
            state = np.asarray(env.mario.game_area()) 
            position = env.mario.level_progress
            state = np.reshape(state, [1, state_size])
            
            i=0
            while feet_val <= 350:
                env.pyboy.tick()
                i += 1
                if i > 60:
                    break
                
            if feet_val >= 350:
                tempo = 2
                for _ in range(tempo):
                    env.pyboy.tick() # Progresses the emulator ahead by one frame.
                
            
            t = 0.0167 * tempo
            time.sleep(t)
            
            fitness = env.mario.fitness
            
            if env.mario.lives_left == 1:
                done = True
                break
            
            time_h += 1
            
        network.lucro = fitness
        
        print('Lucro Total: {}'.format(network.lucro))
        
        #except:
        #network.lucro = 0
        #print('Build failed.')

        env.pyboy.stop()
    return networks

def selection(networks):
    networks = sorted(networks, key=lambda network: network.lucro, reverse=True)
    networks = networks[:int(0.2 * len(networks))]

    return networks


def crossover(networks):
    offspring = []    
    for _ in range(int((population - len(networks)) / 2)):
        parent1 = random.choice(networks)
        parent2 = random.choice(networks)
        child1 = Network()
        child2 = Network()

        # Crossing over parent params
        p1 = int(len(parent1.actions)/2)
        p2 = int(len(parent2.actions)/2)
        a = parent1.actions[:p1]
        b = parent2.actions[:p2]
        new_actions = a + b 
        child1.actions = new_actions
        
        new_actions2 = b + a
        child2.actions = new_actions2

        offspring.append(child1)
        offspring.append(child2)
        
    networks.extend(offspring)
    return networks


def mutate(networks):
    for network in networks[2:]:
        for _ in range(0,int(games/2)):
            val = np.random.uniform(0, 1)
            if val <= 0.2: #mutation chance
                idx = np.random.randint(0,len(network.actions))
                network.actions[idx] = np.random.randint(0,3)

    return networks


def main():
    lucro_nets = []
    best_lucro_nets = []
    best_networks = []
    lucro_nets_media = []
    
    networks = init_networks(population)
    #load best from the past
    try:
        with open('genetic_best_network_mario.json') as json_file:
            data = json.load(json_file)
        lucro = networks[0].set_actions(data.get('actions'),data.get('lucro'))
    except:
        print('Error to load')
        data = None

    for gen in range(generations):
        #env.render()
        print ('Generation {}'.format(gen+1))

        #action
        networks = fitness(networks)
             
        network_lucro = []
        for network in networks:
            network.generation = gen
            network_lucro.append(network.lucro)
            lucro_nets.append(network.lucro)
            if network.lucro > threshold:
                print ('Threshold met')
                print (network.get_actions())
                print ('Best lucro: {}'.format(network.lucro))
                #env.close()
                exit(0)

        #Genetic
        networks = selection(networks)
        #best_networks.append(networks)
        networks = crossover(networks)
        networks = mutate(networks)
                
        print ('Best Fitness: {}'.format(max(network_lucro)))
        best_lucro_nets.append(max(network_lucro))
        #print(network_lucro)
        media = sum(network_lucro)/len(network_lucro)
        print ('Average Fitness: {}'.format(media))
        lucro_nets_media.append(media)
        print("")

    #save best network
    best_net = {'actions': networks[0].get_actions(),'lucro':networks[0].lucro, 'generation':networks[0].generation}
    with open('genetic_best_network_mario.json', 'w') as json_file:
        json.dump(best_net, json_file)
    json_file.close()
                
    plt.subplot(211)
    plt.plot([i for i in range(len(lucro_nets_media))], lucro_nets_media)
    plt.plot([i for i in range(len(best_lucro_nets))], best_lucro_nets)
    plt.ylabel(f"Average Fitness by generation")
    plt.xlabel("Generation #")

    plt.subplot(212)
    plt.plot([i for i in range(len(lucro_nets))], lucro_nets)
    #plt.plot([i for i in range(len(lucro_time))], lucro_time)

    plt.show()
                    

if __name__ == '__main__':
    main()
