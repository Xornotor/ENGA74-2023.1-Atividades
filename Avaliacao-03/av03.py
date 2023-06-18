from time import time
import numpy as np
from numpy.random import rand
import pandas as pd
import plotly.express as px

# 1 - Funções de Benchmark

# 1.1 - Função Esfera
def esfera(x_array, derivada=False):
    if(not derivada):
        return np.sum(np.power(x_array, 2))
    else:
        return 2 * x_array
    
# 1.2 - Função Ackley
def ackley(x_array, derivada=False):
    eps_stability = 1e-8
    exp1 = np.exp(-0.2 * np.sqrt(np.sum(np.power(x_array, 2))/x_array.shape[0]))
    exp2 = np.exp(np.sum(np.cos(2 * np.pi * x_array))/x_array.shape[0])
    
    if(not derivada):
        return (-20 * exp1) - exp2 + 20 + np.e
    else:
        aux1 = np.tile(np.sqrt(np.sum(np.power(x_array, 2)) + eps_stability), (x_array.shape[0], 1)).transpose()
        aux2 = np.divide(x_array, aux1)
        exp1 = np.tile(exp1, (x_array.shape[0], 1)).transpose()
        exp2 = np.tile(exp2, (x_array.shape[0], 1)).transpose()
        coef1 = 2.828
        coef2 = np.pi
        if(x_array.shape[0] == 3):         
            coef1 = 5.6562/np.sqrt(3)
            coef2 = 2 * np.pi / 3
        return (coef1 * aux2 * exp1) + (coef2 * exp2 * np.sin(2 * np.pi * x_array))

# 2 - Funções de treinamento
   
# 2.1 - Gradiente
def gradiente(x_array, funcao, alpha, iteracoes=50):
    x_trained_array = np.copy(x_array)
    func_evolution = [funcao(x_array)]
    time_init = time()
    for _ in range(iteracoes):
        deriv = funcao(x_trained_array, derivada=True)
        x_trained_array = x_trained_array - (alpha * deriv)
        func_evolution.append(funcao(x_trained_array))
    func_evolution = np.array([func_evolution])
    elapsed_time = time() - time_init
    return func_evolution, elapsed_time

# 2.2 - Execução automatizada do Gradiente
def treina_gradiente(alpha, dim, funcao='esfera', iteracoes=50):
    init = (np.random.rand(20, dim) * 60) - 30

    evol_train = np.empty((0, iteracoes + 1))
    times = np.array([])

    if(funcao == 'ackley'):
        f = ackley
    else:
        f = esfera

    for i in range(20):
        func_evolution, elapsed_time = gradiente(init[i], f, alpha, iteracoes)
        evol_train = np.append(evol_train, func_evolution, axis=0)
        times = np.append(times, elapsed_time)

    evol_train=pd.DataFrame(np.transpose(evol_train))
    mean_training_time = np.mean(times) * 1000

    fig = px.line(evol_train, title=f"Treino da função {funcao} em R{dim} com Alpha = {alpha}<br><sup>Tempo médio para {iteracoes} iterações: {mean_training_time:.3f} ms</sup>")
    fig.update_xaxes(title_text='Iterações')
    fig.update_yaxes(title_text=f'Valor da função {funcao}')
    fig.update_layout(legend_title_text='Inicialização')
    fig.show()

# 2.3 - Algoritmo Genético
def genetico(x_matrix, funcao, iteracoes=50, p_recomb=0.1, p_mutacao = 0.01):
    avg_fitness = np.array([])
    min_fitness = np.array([])
    if(x_matrix.shape[1] == 2):
        col = ['x', 'y', 'fitness', 'iter']
    else:
        col = ['x', 'y', 'z', 'fitness', 'iter']
    df_evolution = pd.DataFrame(columns=col)
    x_pop = np.copy(x_matrix)
    time_init = time()
    for i in range(iteracoes):
        # Cálculo de Fitness e inserção no dataframe
        x_fitness = np.array([-funcao(candidato) for candidato in x_pop]).reshape(-1, 1)
        iter_atual = np.tile([i], x_matrix.shape[0]).reshape(-1, 1)
        df_data = np.concatenate((x_pop, x_fitness, iter_atual), axis=1)
        df_evolution = pd.concat([df_evolution, pd.DataFrame(df_data, columns=col)]).reset_index(drop=True)
        # Captura de fitness médio e fitness mínimo por iteração
        avg_fitness = np.append(avg_fitness, np.mean(x_fitness))
        min_fitness = np.append(min_fitness, np.min(x_fitness))
        # Sorting Crescente
        x_sort = np.argsort(x_fitness, axis=0).reshape(-1)
        x_fitness = x_fitness[x_sort]
        x_pop = x_pop[x_sort]
        # Cálculo de probabilidades de seleção
        prob_num = np.array([np.sum(np.arange(1, i+1)) for i in range(1, x_matrix.shape[0]+1)])
        prob_den = np.sum(np.arange(1, x_fitness.shape[0]+1))
        prob = prob_num/prob_den
        # Seleção de elementos
        selecao_prob = np.random.rand(np.ceil(x_fitness.shape[0]/2).astype(np.int32), 2)
        index_selecao_prob = np.searchsorted(prob, selecao_prob, side='right')
        # Recombinação
        new_pop = np.empty((0, x_pop.shape[1]))
        for j in index_selecao_prob:
            candidato1 = x_pop[j[0]]
            candidato2 = x_pop[j[1]]
            recomb_mask = np.random.rand(candidato1.shape[0])
            recomb_mask = np.array([p <= p_recomb for p in recomb_mask])
            novo_candidato1 = np.copy(candidato1)
            np.putmask(novo_candidato1, recomb_mask, candidato2)
            novo_candidato1 = novo_candidato1.reshape((1, -1))
            novo_candidato2 = np.copy(candidato2)
            np.putmask(novo_candidato2, recomb_mask, candidato1)
            novo_candidato2 = novo_candidato2.reshape((1, -1))
            new_pop = np.concatenate((new_pop, novo_candidato1, novo_candidato2), axis=0)
        new_pop = new_pop[:x_pop.shape[0]]
        # Mutação
        mut_gen = (np.random.rand(new_pop.shape[0], new_pop.shape[1]) * 60) - 30
        mut_mask = np.random.rand(new_pop.shape[0], new_pop.shape[1])
        mut_mask = np.array([p <= p_mutacao for p in mut_mask])
        np.putmask(new_pop, mut_mask, mut_gen)
        # Nova População
        x_pop = np.copy(new_pop)

    elapsed_time = (time() - time_init) * 1000
    fitness_metrics = np.concatenate(([avg_fitness], [min_fitness]), axis = 0).transpose()
    df_fitness = pd.DataFrame(fitness_metrics)
    #df_fitness['Aptidao'] = ['Media', 'Minima']
        
    return df_evolution, df_fitness, elapsed_time

# 2.4 - Execução automatizada do Algoritmo Genético
def treina_genetico(dim=2, funcao='esfera', iteracoes=50, taxa_recomb=0.1, taxa_mut=0.01):
    init = (np.random.rand(20, dim) * 60) - 30
        
    if(funcao == 'ackley'):
        f = ackley
        range_color = [-100, 0]
    else:
        f = esfera
        range_color = [-1500, 0]

    range_xyz = [-30, 30]
    
    df_evol, df_fit, train_time = genetico(init, f, iteracoes=iteracoes,
                                           p_recomb=taxa_recomb, p_mutacao=taxa_mut)    

    if(dim == 3):
        fig1 = px.scatter_3d(df_evol, x='x', y='y', z='z', color='fitness', animation_frame='iter',
                        color_continuous_scale=px.colors.sequential.YlGnBu,
                        range_y=range_xyz, range_x=range_xyz, range_z=range_xyz, range_color=range_color,
                        title=f"")
    else:
        fig1 = px.scatter(df_evol, x='x', y='y', color='fitness', animation_frame='iter',
                        color_continuous_scale=px.colors.sequential.YlGnBu,
                        range_y=range_xyz, range_x=range_xyz, range_color=range_color,
                        title=f"Evolução da população para função {funcao} em R{dim} | Recomb. = {taxa_recomb * 100}% | Mut. = {taxa_mut * 100}%<br><sup>Tempo gasto para {iteracoes} iterações: {train_time:.3f} ms</sup>")

    fig2 = px.line(df_fit, category_orders={"variable": ["Média", "Mínima"]},
                   title=f"Evolução da aptidão para função {funcao} em R{dim} | Recomb. = {taxa_recomb * 100}% | Mut. = {taxa_mut * 100}%<br><sup>Tempo gasto para {iteracoes} iterações: {train_time:.3f} ms</sup>")
    
    for idx in enumerate(fig2["data"]):
        if(idx[0] == 0):
            idx[1]["name"] = "Média"
        elif(idx[0] == 1):
            idx[1]["name"] = "Mínima"
        
    fig2.update_xaxes(title_text='Iterações')
    fig2.update_yaxes(title_text='Aptidão')
    fig2.update_layout(legend_title_text='Aptidão')
    
    fig1.show()
    fig2.show()

# 2.5 - Enxame de Partículas
def enxame():
    pass

# 2.4 - Execução automatizada do Enxame de Partículas
def treina_enxame():
    pass