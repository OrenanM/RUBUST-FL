import torch
import os
import numpy as np
import h5py
import copy
import time
import random

import sys
import pandas as pd
import collections

from utils.data_utils import read_client_data, load_validate_server
from utils.dlg import DLG

from sklearn.cluster import SpectralClustering
from flcore.cluster.cudacka import CudaCKA
from flcore.cluster.cka import CKA
from flcore.clients.clientmaliciousavg import ClientMaliciousAVG
from flcore.clients.clientmaliciousala import ClientMaliciousALA


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        self.uploaded_models2 = []
        self.uploaded_clients = {}

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch = args.fine_tuning_epoch

        #atributos novos
        self.clients_weigths = []
        self.current_round = -1
        self.cluster = args.cluster
        self.num_clusters = args.num_clusters
        self.clients_cluster = dict()
        self.entropy = args.entropy
        self.type_select = args.type_select
        self.weigth_size_entropy = args.weigth_size_entropy
        self.client_fake = args.client_fake
        self.num_client_fake = args.num_client_fake
        self.remove_cf = args.remove_cf
        self.substutive_client_fake = args.substutive_client_fake
        self.probabity_cf = args.probabity_cf

        self.n_clients_fake = 0
        self.n_removed = 0
        self.n_tp_removed = 0

        self.rate_client_fake = args.rate_client_fake
        self.atack = args.atack
        self.srn_noise = args.srn_noise

        self.list_global_model = []

        self.tot_sample_selected = 0

        self.n_client_malicious = args.n_client_malicious
        self.quarantine = args.quarantine
        self.round_init_atk = args.round_init_atk

        self.sc = args.score_cluster

    def set_clients(self, clientObj):
        indexes = list(range(self.num_clients))
        n_malicious = self.n_client_malicious

        index_malicious = np.random.choice(indexes, n_malicious)

        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            
            if i not in index_malicious:
                client = clientObj(self.args, id=i, train_samples=len(train_data), 
                                test_samples=len(test_data), train_slow=train_slow, 
                                send_slow=send_slow)
            elif i in index_malicious and self.algorithm == "FedAvg":
                client = ClientMaliciousAVG(self.args, id=i, train_samples=len(train_data), 
                                test_samples=len(test_data), train_slow=train_slow, 
                                send_slow=send_slow)
            elif i in index_malicious and self.algorithm == "FedALA":
                client = ClientMaliciousALA(self.args, id=i, train_samples=len(train_data), 
                                test_samples=len(test_data), train_slow=train_slow, 
                                send_slow=send_slow)
                
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    
# ***********************************************************************************************************************************
    def save_results_txt(self, acc=0, loss=0, frr=0, fpr=0, interrupt = False):
        #self.n_clients_fake = self.n_client_malicious
        type_sel = self.type_select
        result_path = 'result2'

        #cria o diretório de resultados
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        #cria o diretório de resultados de cada seleção
        sel_path = os.path.join(result_path, type_sel)
        if not os.path.exists(sel_path):
            os.mkdir(sel_path)

        name_file = f"{self.dataset}_nclients{self.num_clients}_jr{self.join_ratio}_" + \
                    f"remove{self.remove_cf}_nmalicious{self.n_client_malicious}" + \
                    f"_ncl{self.num_clusters}_atk_{self.atack}_srn_({self.srn_noise})" + \
                    f"{self.algorithm}_quarantine{self.quarantine}" + \
                    f"init_{self.round_init_atk}_score_{self.sc}.txt"
        path_file = os.path.join(sel_path, name_file)

        #cria o arquivo txt caso não exista
        if interrupt:
            with open(path_file, "a") as arquivo:
                arquivo.write('0, 0, 0, 0, 0, 0, 0\n')
                return

        if (not os.path.exists(path_file)) or (self.times == 0) and (self.current_round == 0):
            with open(path_file, "w") as arquivo:
                arquivo.write(f"acuracia,loss,frr,fpr, removed, n_clf, tp_removed\n{acc}, {loss}, {frr}, {fpr}, {self.n_removed},{self.n_clients_fake}, {self.n_tp_removed}\n")
        else:
            with open(path_file, "a") as arquivo:
                arquivo.write(f"{acc}, {loss}, {frr}, {fpr}, {self.n_removed},{self.n_clients_fake}, {self.n_tp_removed}\n")
    
    def calculate_entropy(self):
        entropies = np.array([client.client_entropy() for client in self.clients])
        entropies[np.isnan(entropies)] = 0

        for client, entropy in zip(self.clients, entropies):
            client.entropy = entropy

    def select_clients(self):
        self.current_round += 1
        self.fit_select = self.not_in_quarantine()
        self.sub_quarantine()

        if self.current_round == 0 and self.entropy == 1: #calcula a entropia
            self.calculate_entropy()
        
        #seleciona o tipo de seleção
        if self.type_select == "A":
            selected_clients = self.select_random()
        elif self.type_select == "B":
            selected_clients = self.select_entropy_size()
        elif self.type_select == "C":
            selected_clients = self.select_entropy()
        elif self.type_select == "D":
            selected_clients = self.select_size()
        elif self.type_select == "E":
            selected_clients = self.select_entropy_size_polynomial()

        return selected_clients

    def select_random(self):
        '''Seleciona os clientes de forma aleatória'''
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)
        else:
            self.current_num_join_clients = self.current_num_join_clients
        
        selected_clients = list(np.random.choice(self.fit_select, self.current_num_join_clients, replace=False))
        return selected_clients
    
    def select_entropy_size_polynomial(self, num_select=0, select = None):
        """
        Aplica pesos considerando a entropia e o tamanho do conjunto de dados, 
        aumentando a distância entre esses pesos por meio de uma função polinomial 
        quíntupla, como f(x) = x⁵.

            -Seleciona 2 vezes jr clientes com pesos proporcionais ao tamanho do conjunto 
            de dados, ajustando a distância com a função polinomial quíntupla.
            -Em seguida, escolhe jr clientes adicionais com pesos determinados pela entropia, 
            também ajustando a distância com a função polinomial quíntupla.
        """
        if select is None:
            select = self.fit_select
            num_select = round(len(select) * self.join_ratio)
        num_select_size = 2 * num_select

        #Calula os pesos do dataset
        weigth_size = np.array([client.train_samples for client in select])
        weigth_size = weigth_size/weigth_size.sum()

        #Seleciona com peso dos tamanho do dataset
        clients_select_size = np.random.choice(select, num_select_size, 
                                               p = weigth_size, replace=False)

        #Calcula os pesos da entropia
        weigth_entropy = np.array([client.entropy for client in clients_select_size])
        weigth_entropy = weigth_entropy/weigth_entropy.sum()

        #Seleciona com peso da entropia
        clients_select_entropy = np.random.choice(clients_select_size, num_select,
                                                  p=weigth_entropy, replace=False)
        
        return clients_select_entropy
    
    def select_entropy_size(self):
        """
        Seleciona clientes com base em uma porcentagem (jr) do total de clientes,
        aplicando pesos proporcionais ao tamanho do conjunto de dados e à entropia.

        Seleciona 2 vezes jr clientes com pesos proporcionais ao tamanho do conjunto de dados.
        Em seguida, escolhe jr clientes adicionais com pesos determinados pela entropia.

        """
        select = self.fit_select
        num_select_size = round(len(select) * self.join_ratio * 2)
        num_select_entropy = round(len(select) * self.join_ratio)

        #Calula os pesos do dataset
        weigth_size = np.array([client.train_samples for client in select])
        weigth_size = weigth_size/weigth_size.sum()

        #aplica a função polinomial
        weigth_size_ajusted = weigth_size ** 5

        weigth_size_ajusted = weigth_size_ajusted/sum(weigth_size_ajusted)
        weigth_size_ajusted = weigth_size_ajusted/sum(weigth_size_ajusted)

        #Seleciona com peso dos tamanho do dataset
        clients_select_size = np.random.choice(self.clients, num_select_size, 
                                               p = weigth_size_ajusted, replace=False)

        #Calcula os pesos da entropia
        weigth_entropy = np.array([client.entropy for client in clients_select_size])
        weigth_entropy = weigth_entropy/weigth_entropy.sum()

        #aplica a função polinomial
        weigth_entropy_ajusted = weigth_entropy ** 5 

        weigth_entropy_ajusted = weigth_entropy_ajusted/sum(weigth_entropy_ajusted)
        weigth_entropy_ajusted = weigth_entropy_ajusted/sum(weigth_entropy_ajusted)

        #Seleciona com peso da entropia
        clients_select_entropy = np.random.choice(clients_select_size, num_select_entropy,
                                                  p=weigth_entropy_ajusted, replace=False)
        
        return clients_select_entropy
    
    def select_entropy(self):
        "Seleciona os clientes com pesos proporcionais as suas entropias"

        num_select_entropy = round(len(self.fit_select) * self.join_ratio)
        sort_entropy = sorted(self.fit_select, key = lambda client: client.entropy, reverse=True)
        clients_select_size = sort_entropy[:num_select_entropy]
        return clients_select_size
    
    def select_size(self):
        "Seleciona os clientes com pesos proporcionais ao tamanho de seus datasets"
        num_select_size = round(len(self.fit_select) * self.join_ratio)
        
        sort_size = sorted(self.fit_select, key = lambda client: client.train_samples, reverse=True)
        clients_select_size = sort_size[:num_select_size]
        print(f"selecionados: {[client.id for client in clients_select_size]}")
        return clients_select_size
        
    def print_cluster_clientes(self):
        '''Mostra o dataframe que relacionado clusters e clientes'''
        
        data = {'Cluster': [], 'Clientes': []}
        
        for label, clients in self.clients_cluster.items():
            data['Cluster'].append(label)
            data['Clientes'].append(', '.join(str(client.id) for client in clients))

        df = pd.DataFrame(data)
        df = df.sort_values(by='Cluster')
        print(df.to_string(index=False))

        #exibe o score de cada cluster
        score_sort = {chave: valor for chave, valor in 
                      sorted(self.score_cluster.items())}
        
        for cluster, score in score_sort.items():
            print(f'cluster {cluster} - score: {score}')
    
    def calculate_cka2(self):
        X = self.uploaded_models

        cka_calculator = CudaCKA(self.device)

        row_indices, col_indices = np.triu_indices(len(X))

        matriz_similaridade_cka = np.zeros((len(X), len(X)))
        #data_representations = load_dataset_representations(self.dataset, self.batch_size, self.num_classes)
        data_representations = self.clients[0].load_test_data()

        
        for i, j in zip(row_indices, col_indices):

            similaridade_cka = cka_calculator.calcule_cka(X[i], X[j], data_representations,
                                                          self.algorithm)
            
            # Preenche a matriz simétrica com os valores
            matriz_similaridade_cka[i, j] = similaridade_cka
            matriz_similaridade_cka[j, i] = similaridade_cka
        

        df_similaridade = pd.DataFrame(matriz_similaridade_cka, 
                                       columns = self.uploaded_ids,
                                       index= self.uploaded_ids)
        
        if self.sc == 1:
            return matriz_similaridade_cka
        elif self.sc == 2:
            df_similaridade.fillna(0., inplace = True)
            return df_similaridade

    def calculate_cka(self):
        '''Realiza o calcula de similaridade'''
        clients_weights = []

        for model in self.uploaded_models:
            weights = [param.data for param in model.parameters()]
            flattened_weights = torch.cat([w.view(-1) for w in weights])
            clients_weights.append(flattened_weights)

        X = clients_weights
        
        cka_calculator = CKA(self.device)
        row_indices, col_indices = np.triu_indices(len(X))

        matriz_similaridade_cka = np.zeros((len(X), len(X)))

        for i, j in zip(row_indices, col_indices):
            X_i = X[i].reshape(-1, 1)
            X_j = X[j].reshape(-1, 1)
            similaridade_cka = cka_calculator.linear_CKA(X_i, X_j)
            
            # Preenche a matriz simétrica com os valores
            matriz_similaridade_cka[i, j] = similaridade_cka
            matriz_similaridade_cka[j, i] = similaridade_cka
            
        matriz_similaridade_cka = np.nan_to_num(matriz_similaridade_cka)
        return matriz_similaridade_cka
    
    def set_clients_cluster(self):
        '''Cria um dicionario com clusters e clientes'''
        clients_cluster = {chave: [] for chave in self.labels}
        labels = list(set(self.labels))
        
        for label in labels:
            for client in self.selected_clients:
                if client.cluster == label:
                    clients_cluster[label].extend([client])
        
        self.clients_cluster = clients_cluster
    
    def cluster_cka(self):
        '''Realiza a clusterização do cka'''

        if self.sc == 1:
            matriz_similaridade_cka = self.calculate_cka()
        else:
            matriz_similaridade_cka = self.calculate_cka2()
        
        cka_cluster = SpectralClustering(n_clusters = self.num_clusters, affinity='precomputed')
        self.labels = cka_cluster.fit_predict(matriz_similaridade_cka)

        for client, label in zip(self.selected_clients, self.labels):
            client.cluster = label
        
        self.set_clients_cluster()
        
        if self.sc == 1:
            self.calculate_score_cluster(matriz_similaridade_cka)
        elif self.sc == 2:
            df_similaridade = pd.DataFrame(matriz_similaridade_cka)
            self.calculate_score_cluster2(df_similaridade)

        #self.calculate_score_cluster3()

    def calculate_score_cluster3(self):
        self.score_cluster = {}
        #self.print_cluster_clientes()
        n_removed = 0
        similarity_global = self.calculate_simalality_delay_models(self.global_model)
        print(f"global {similarity_global}")
        for label, cluster in self.clients_cluster.items():
            ids_cluster = [client.id for client in cluster]
            indexes_cluster = [self.uploaded_ids.index(id) for id in ids_cluster]

            uploaded_weights = np.array(self.uploaded_weights)[indexes_cluster]
            uploaded_models = np.array(self.uploaded_models)[indexes_cluster]
            
            tot_samples = 0

            for client in cluster:
                tot_samples += client.train_samples
        
            for i, w in enumerate(uploaded_weights):     
                uploaded_weights[i] = (w * self.tot_sample_selected) / tot_samples

            cluster_model = self.aggregate_cluster(uploaded_models, uploaded_weights)
            list_similarity = self.calculate_simalality_delay_models(cluster_model)

            print(f'label - {label}:')
            print(f"similaridade: {list_similarity}")

            #list_similarity = np.array(list_similarity)

            '''n = len(list_similarity) 

            if n > 0:
                n = len(list_similarity) - 5
                list_similarity = list_similarity[:n]
            else:
                list_similarity = []'''

            #similar_ant = sum(list_similarity > 0.95)
            similar_ant = sum(1 for x in list_similarity if x > 0.98)

            if similar_ant:
                self.score_cluster[label] = 1
                n_removed += 1
            else:
                self.score_cluster[label] = 0

        if n_removed == 0:
            for label, cluster in self.clients_cluster.items():
                self.score_cluster[label] = 1

    def calculate_simalality_delay_models(self, model_cluster):
        similarity_list = []

        cka_calculator = CudaCKA(self.device)
        data_representations = self.clients[0].load_test_data()
        
        for global_model in self.list_global_model:
            similarity = cka_calculator.calcule_cka(model_cluster, 
                                                    global_model, 
                                                    data_representations)
            similarity_list.append(similarity)
        return similarity_list
    
    def test_metrics_cluster(self, model):
        testloaderfull = load_validate_server(dataset=self.dataset, size_batch=self.batch_size)
        model.eval()

        test_acc = 0
        n_data = 0
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                n_data += len(y)
        test_acc = test_acc/n_data
        return test_acc
            
    def calculate_score_cluster2(self, matriz_similaridade):
        self.score_cluster = {}
        #print(matriz_similaridade)
        #self.print_cluster_clientes()

        maior_chave = max(self.clients_cluster, key=lambda k: len(self.clients_cluster[k]))
        maior_lista = self.clients_cluster[maior_chave]

        for cluster_label in self.clients_cluster.keys():
            self.score_cluster[cluster_label] = 0
            min_similidade = 1
            for client in self.clients_cluster[cluster_label]:
                client_id = client.id
                client_similaridades = matriz_similaridade.loc[matriz_similaridade.index != client_id, client_id]
                mean_client = len(client_similaridades)/sum(1/(client_similaridades))
                #print(f'cluster: {cluster_label}, mean: {mean_client}')

                if mean_client < min_similidade:
                    min_similidade = mean_client

            if cluster_label == maior_chave:
                self.score_cluster[cluster_label] = 1
            else:
                self.score_cluster[cluster_label] = min_similidade

        #print(self.score_cluster)

        return self.score_cluster


    def calculate_score_cluster(self, matriz_similaridade):
        '''Calcula uma pontuação para cada cluster considerando a média de similaridade 
        entre os clientes do cluster'''
        self.score_cluster = {}
        #matriz_similaridade = matriz_similaridade.values
        
        for cluster_label, clientes in self.clients_cluster.items():
            # Identifica os índices dos clientes neste cluster
            indices_clientes = [cliente.id for cliente in clientes]  # cada cliente tem um atributo 'index'
            indices_clientes = [self.uploaded_ids.index(id) for id in indices_clientes]
            # Calcula a média das similaridades para este cluster
            soma_similaridades = 0
            num_pares = 0
            for i in range(len(indices_clientes)):
                for j in range(i + 1, len(indices_clientes)):  # Evita comparar o cliente com ele mesmo e duplicatas
                    soma_similaridades += matriz_similaridade[indices_clientes[i]][indices_clientes[j]]
                    num_pares += 1
                              
            # Evita divisão por zero se o cluster tiver menos de 2 clientes
            if num_pares > 0:
                media_similaridade = soma_similaridades / num_pares
            else:
                media_similaridade = 0      
            
            # Armazene a pontuação média para este cluster
            self.score_cluster[cluster_label] = media_similaridade
            
        return self.score_cluster
    
# ***********************************************************************************************************************************
    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

        self.list_global_model.append(self.global_model)

    def add_quarantine(self, clients):
        for client in clients:
            client.negative_score = 2 * client.negative_score
            client.rounds_quarentine += client.negative_score

    def sub_quarantine(self):
        for client in self.clients:
            if client.rounds_quarentine > 0:
                client.rounds_quarentine -= 1

    def exibe_quarantine(self):
        quarantine_clients = []
        for client in self.clients:
            if client.rounds_quarentine > 0:
                quarantine_clients.append(client)
        
        for client in quarantine_clients:
            id = client.id
            rounds_quarantine = client.rounds_quarentine
            print(f'Client {id}: {rounds_quarantine} rounds quarantine')
    
    def not_in_quarantine(self):
        quarantine_clients = []
        for client in self.clients:
            if client.rounds_quarentine == 0:
                quarantine_clients.append(client)
        
        return quarantine_clients
    
    def remove_client_fake(self):
        
        self.print_cluster_clientes()
        # Cria uma cópia dos clientes selecionados
        selected = self.selected_clients[:]
        
        # Identifica os clusters a serem removidos com base na pontuação
        remove_clusters = [label for label, score in self.score_cluster.items() if score < 0.02]
        
        # Filtra os clientes a serem removidos com base nos clusters e se estão na lista selecionada
        remove_clients = [client for client in selected if client.cluster in remove_clusters]
        print(f'Selecionados antes da remoção: {[client.id for client in self.selected_clients]}')
        # Remove os clientes marcados para remoção

        self.n_tp_removed = sum([client.is_malicious for client in remove_clients])
        self.n_clients_fake = sum([client.is_malicious for client in self.selected_clients])
        self.n_removed = len(remove_clients)
        if self.quarantine == 1:
            self.add_quarantine(remove_clients)
            self.exibe_quarantine()

        print(f'removidos certo: {self.n_tp_removed}')
        print(f'clientes falsos: {self.n_clients_fake}')
        print(f'removidos: {self.n_removed}')
        
        for client in remove_clients:
            print(f'Removed: {client.id}')
            self.selected_clients = list(self.selected_clients)
            self.remove_client_update(client)
            

        print(f'Selecionados: {[client.id for client in self.selected_clients]}')
        print(f'Porcentagem de clientes selecionados: {(len(self.selected_clients)/self.num_clients * 100)} %')
        num_remove = len(remove_clients)
        if num_remove > 0 and self.substutive_client_fake:
            self.replace_client(num_substitutive = num_remove)
        
    def replace_client(self, num_substitutive):
        #seleciona
        not_selected = [client for client in self.clients if client not in self.selected_clients]
        selected_substitute = self.select_entropy_size_polynomial(select=not_selected, 
                                                                  num_select=num_substitutive)
        self.selected_clients.extend(selected_substitute)
        #treina
        for client in selected_substitute:
            client.train()

        #clusteriza
        self.cluster_cka()

        #remove 
        self.remove_client_fake()
        print(f'Selecionados após substituição: {[client.id for client in self.selected_clients]}')

    #seleciona -> treina -> recebe -> clusteriza -> remove -> substui -> agrega
    #substituição:
    #seleciona -> treina -> recebe -> clusteriza -> remove (se não remover ninguem termina)
    def remove_client_update(self, client):

        self.selected_clients.remove(client)

        index_remove = self.uploaded_ids.index(client.id)

        self.uploaded_models.remove(self.uploaded_models[index_remove])
        self.uploaded_ids.remove(client.id)
        active_clients = self.selected_clients

        self.uploaded_weights = []
        tot_samples = 0

        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_weights.append(client.train_samples)

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

        return active_clients
    
    def receive_models(self):
        
        assert (len(self.selected_clients) > 0)

        active_clients = self.selected_clients

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0

        for n, client in enumerate(active_clients):
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
                
            except ZeroDivisionError:
                client_time_cost = 0

            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.send_local_model(self.current_round))
                #print(f"selected: {client.id}")

                # dicionario para mapeadar pesos e modelos pelo id do cliente
                self.uploaded_clients[client.id] = (self.uploaded_weights[n], 
                                                    self.uploaded_models[n])

        self.tot_sample_selected = tot_samples

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
        
        #self.n_clients_fake = sum([x.send_fake for x in self.selected_clients])

        return active_clients
    
    def aggregate_cluster(self, uploaded_models_cluster, uploaded_weights_cluster):
        """
        agrega os modelos de cada cluster 
        """
        assert (len(uploaded_models_cluster) > 0)

        local_model = copy.deepcopy(uploaded_models_cluster[0])
        for param in local_model.parameters():
            param.data.zero_()
    
        for w, client_model in zip(uploaded_weights_cluster, uploaded_models_cluster):
            self.add_parameters_cluster(w, client_model, local_model)

        return local_model

    def add_parameters_cluster(self, w, client_model, global_model):
        """
        adiciona os modelos da cada cluster
        """
        for server_param, client_param in zip(global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
              
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
    
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
        
        #self.calculate_simalality_delay_models()
        
    
    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w
        

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
 
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []

        fp_rate = []
        fr_rate = []

        losses = []

        for c in self.clients:
            ct, ns, auc, fpr, frr, cl = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

            fp_rate.append(fpr)
            fr_rate.append(frr)

            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc, fp_rate, fr_rate, losses

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None, nnc = 0):
        stats = self.test_metrics()
        stats_train = self.train_metrics()
        
        test_frr = sum(stats[5]) * 1.0 / len(stats[5]) #mean frr
        test_fpr = sum(stats[4]) * 1.0 / len(stats[4]) #mean fpr

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        test_loss = sum(stats[6])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        self.save_results_txt(test_acc, test_loss, test_frr, test_fpr)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Loss: {:.4f}".format(test_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        print(f'False Rejection Rate: {test_frr:.4f}')
        print(f'False Positive Rate: {test_fpr:.4f}')

        #if (self.cluster != None and self.current_round>0):
        #    self.print_cluster_clientes()

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc 