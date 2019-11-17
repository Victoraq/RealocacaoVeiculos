#!/usr/bin/env python
# coding: utf-8


# Grupo: Bryan Barbosa, Gabriele Iwashima e Victor Aquiles Alencar


from gurobipy import *
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
import getopt
import sys
import os.path
import csv
import warnings
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.nx_pydot import write_dot
warnings.filterwarnings("ignore")

grafo = nx.Graph()
pos =graphviz_layout(grafo, prog='dot')

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate distance between the geo coordinates
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    m = 6367 * c * 1000

    return m


def geojson_builder(down_left,up_right,travels=0):
    """
    Build a rectangle geojson from two points.
    """
    p1 = [str(down_left[0]),str(down_left[1])]
    p2 = [str(up_right[0]),str(down_left[1])]
    p3 = [str(up_right[0]),str(up_right[1])]
    p4 = [str(down_left[0]),str(up_right[1])]
    
    geojson = {"type": "FeatureCollection",
              "features": [
                {"type": "Feature",
                    "properties":{
                        "arrivals_outs": travels
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[p1,p2,p3,p4,p1]] }}]}

    return geojson


def build_regions(data, points):
    """
    Create a list of geojsons that represent the group more 
    significant in the region.
    """
  
    # Creating matrix of the region coordinates
    # matrix dimensions based in points
    lat = points['lat'].unique()
    step = round(abs(lat[0]-lat[1]),5)
    diff = abs(lat.max() - lat.min())

    len_y = int(diff/step)+1

    lon = points['lon'].unique()
    step = round(abs(lon[0]-lon[1]),5)
    diff = abs(lon.max() - lon.min())

    len_x = int(diff/step)+1
    
    
    matrix = []

    # filling the matrix with the points coordinates
    c = 0
    for y in range(len_y):
        line = []
        for x in range(len_x):
            line.append((points['lon'].iloc[c], points['lat'].iloc[c]))
            c+=1
        matrix.append(line)

    geojsons = []
    
    # Column options for travels and idle datasets
    start_lat = 'Start_lat'
    start_lon = 'Start_lon'
    end_lat = 'End_lat'
    end_lon = 'End_lon'
    
    for lin in range(len(matrix)-1):
        for col in range(len(matrix[0])-1):

            # Selecting all points inside a square of matrix coordinates
            down_left = matrix[lin][col]
            up_right = matrix[lin+1][col+1]
            
            start_lon_condition = (data[start_lon] >= down_left[0]) & (data[start_lon] <= up_right[0])
            start_lat_condition = (data[start_lat] >= down_left[1]) & (data[start_lat] <= up_right[1])
            
            end_lon_condition = (data[end_lon] >= down_left[0]) & (data[end_lon] <= up_right[0])
            end_lat_condition = (data[end_lat] >= down_left[1]) & (data[end_lat] <= up_right[1])

            start_travels = data.loc[start_lon_condition & start_lat_condition]
            end_travels = data.loc[end_lon_condition & end_lat_condition]

            # Middle of the region
            middle_lat = (down_left[1] + up_right[1])/2
            middle_lon = (down_left[0] + up_right[0])/2

            # Build a geojson
            geojsons.append(geojson_builder(down_left,up_right, len(start_travels)+len(end_travels)))

            # Changing start region and end region of the travels
            data.loc[start_lon_condition & start_lat_condition, 'start_region'] = len(geojsons)-1
            data.loc[start_lon_condition & start_lat_condition, start_lat] = middle_lat
            data.loc[start_lon_condition & start_lat_condition, start_lon] = middle_lon

            data.loc[end_lon_condition & end_lat_condition, 'end_region'] = len(geojsons)-1
            data.loc[end_lon_condition & end_lat_condition, end_lat] = middle_lat
            data.loc[end_lon_condition & end_lat_condition, end_lon] = middle_lon
            
    data.dropna(axis=0, inplace=True)

            
    return geojsons


def limpeza(data):
    """
    Realiza limpeza padrão dos dados.
    """


    # only travels with more than 30 minutes of duration, that is the limit of cancellation of a reserve
    data = data[(data['duration'] > 30) | (data['distance'] > 3)]

    # preco real das viagens
    data['price'] = data['duration'] * 0.41

    data.Start_time = pd.to_datetime(data.Start_time)
    data.End_time = pd.to_datetime(data.End_time)

    # Colleting vehicle ids
    car_ids = list(data.Id.unique())

    # Removing uncommon ids
    # Ex: 4c5865a3-4b03-40f6-a3a8-d4e94aae3b17
    ids_uncommon = [id for id in car_ids if id.find('-') != -1]
    car_ids = [id for id in car_ids if id.find('-') == -1]

    data = data[~data.Id.isin(ids_uncommon)]
    
    data.reset_index(inplace=True, drop=True)
    
    return data


def get_data(path, time_space='m', time_interval='1-w'):
    """
        Retorna dados por meio de um dataframe e sua lista de regiões.

        :parameter path: caminho dos dados a serem lidos

        :parameter time_space: indica a granulariedade do tempo, m->minutos, h->hora

        :parameter time_interval: indica o intervalo de tempo dos dados no formato: n-p,
                                  p - indica a granulariedade: d->dias, w->semanas e m->meses
                                  n - indica a quantidade de intervalos de tempo
                                  Ex: 5-w, irá ler dados de 5 semanas
    """
    data = pd.read_csv(path)
    #data = data.loc[:200]
    print('Limpando dados...')
    data = limpeza(data) # limpando dados
    
    assert time_space == 'h' or time_space == 'm'
    print('Selecionando espaço de tempo...')
    # selecionando se o espaço de tempo será em horas ou minutos
    if time_space == 'h':
        data.Start_time = data.Start_time.apply(lambda x: x.replace(minute=0,second=0, microsecond=0))
        data.End_time = data.End_time.apply(lambda x: x.replace(minute=0,second=0, microsecond=0))
    elif time_space == 'm':
        data.Start_time = data.Start_time.apply(lambda x: x.replace(second=0, microsecond=0))
        data.End_time = data.End_time.apply(lambda x: x.replace(second=0, microsecond=0))
    
    print('Selecionando periodo de coleta...')
    # selecionando periodo a ser analisado
    value, period = time_interval.split('-')
    
    assert period in ['d','w','m']
    
    if period == 'd': mult = 1
    elif period == 'w': mult = 7
    elif period == 'm': mult = 30
        
    inicio = data.Start_time[0]
    fim = inicio + timedelta(days=int(value) * mult)
    
    data = data[data.End_time <= fim]
    
    print('Criando regiões...')
    # Creating points of Vancouver region
    latitudes = []
    longitudes = []
    step =0.004
    for lat in np.arange(data['Start_lat'].min(),data['Start_lat'].max()+step,step):
        for lon in np.arange(data['Start_lon'].min(),data['Start_lon'].max()+step,step):
            latitudes.append(lat)
            longitudes.append(lon)

    points = pd.DataFrame({'lon':longitudes, 'lat':latitudes})
    
    # estabelecendo e adicionando coluna de regioes
    regions = build_regions(data, points)
    
    # removendo regioes sem viagens
    regions = [r for r in regions if r['features'][0]['properties']['arrivals_outs'] != 0]
    regions.insert(0,None)

    # Adding a distance(m) column
    data['distance'] = data.apply(lambda x: haversine(x.Start_lon,x.Start_lat,x.End_lon,x.End_lat), axis=1)

    data.reset_index(inplace=True, drop=True)
        
    return data, regions


def calcula_custo_realocacao(viagens, viagem_id, custo_realocacao, custo_real, local_realoc, bonus):
    """
    Calcula custo da viagem dado o bonus e sua realocação
    """
    viagem = viagens[viagem_id]

    custo_viagem = custo_real[(viagem[0],viagem[1])]

    if local_realoc is None:
        custo_realc = 0
    else:
        custo_realc = custo_realocacao[(viagem[1],local_realoc)]

    return (custo_viagem - bonus*(custo_realc))

def main(argv):

    path = 'data/evo_travels.csv'
    model_lp = None
    time_space='m'
    time_interval='1-d'
    porcentagem = 0.2
    realocacao = True
    bonus = 0.3

    try:
      opts, args = getopt.getopt(argv,"hi:d:h:m:g:i:p:r:b:",["datafile=","granularidade=",
                                               "intervalo=","porcentagem=","realocacao=","bonus="])
    except getopt.GetoptError:
        print(
            'modelo.py -d <datafile> -m <modelo> -g <granularidade> -i <intervalo> -p <porcentagem> -r <realocacao> -b <bonus>'
            )
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(
                'modelo.py -d <datafile> -m <modelo> -g <granularidade> -i <intervalo> -p <porcentagem> -r <realocacao> -b <bonus>'
                )
            sys.exit()
        elif opt in ("-d", "--datafile"):
            path = arg
        elif opt in ("-m", "--modelo"):
            modelo_lp = arg
        elif opt in ("-g", "--granularidade"):
            time_space = arg
        elif opt in ("-i", "--intervalo"):
            time_interval = arg
        elif opt in ("-p", "--porcentagem"):
            porcentagem = float(arg)
        elif opt in ("-r", "--realocacao"):
            if arg == 'True': realocacao = True
            else: realocacao = False
        elif opt in ("-b", "--bonus"):
            bonus = float(arg)


    print(f"""Parâmetros escolhidos:
            datafile = {path},
            model_lp = {model_lp},
            granularidade = {time_space},
            intervalo = {time_interval},
            porcentagem = {porcentagem},
            realocacao = {realocacao},
            bonus = {bonus}""")


    if time_space == 'h': custo_fixo = 0.41*60
    else: custo_fixo = 0.41

    t_inicial = datetime.now()
    
    # leitura dos dados
    data, regions = get_data(path, time_space, time_interval) 
    

    # ### Localizando viagens por regiões em vez de coordenadas


    # Creating points of Vancouver region
    latitudes = []
    longitudes = []
    step =0.004 # pontos com espaçamento de 600x600 metros
    for lat in np.arange(data['Start_lat'].min(),data['Start_lat'].max()+step,step):
        for lon in np.arange(data['Start_lon'].min(),data['Start_lon'].max()+step,step):
            latitudes.append(lat)
            longitudes.append(lon)

    points = pd.DataFrame({'lon':longitudes, 'lat':latitudes})


    # Dados que serão utilizados no modelo

    # Todos os instantes de tempo em timestamp
    time_instants = pd.date_range(start=data.Start_time.min(), end=data.End_time.max(), freq='h') 
    viagem_id = data.index.values                   # id de todas as viagens
    n_vehicles = int(len(data.Id.unique()) * porcentagem)   # numero total de veículos
    travel_possibilities = set()                    # possibilidades de viagens
    custo_realocacao = {}                              # custo de cada viagem
    custo_real = {}

    # preenchendo possibilidades de viagens e seus custos
    for t in map(tuple,data[['start_region','end_region','distance', 'price']].values):
        travel_possibilities.add((t[0], t[1]))

        if t[0] != t[1]: # não existem realocações para o mesmo local
            # custo da realocacao em relacao a distancia e assumindo uma velocidade de 60 km/h
            custo_realocacao[(t[0],t[1])] = (t[2] / 1000) * 0.41  # metros/min
            
        # custo da viagem é o custo real do dataset
        custo_real[(t[0],t[1])] = t[3]
    
    print('Montando o modelo...')
    m = Model("realocacao")

    if model_lp == None:
        # Variável que indica se ocorreu a viagem com realocação para p
        print('Variável que indica se ocorreu a viagem com realocação para p')
        locais = [None] + list(data.start_region.unique())
        viagens_realizadas = {}
        nome_viagem_cr = []
        nome_viagem_sr = []
        for v in viagem_id:
            #print('Quantas viagens faltam:',len(viagem_id) - c)
            #print('Tamanho do vetor de x:', len(viagens_realizadas))
            end_region = data.end_region.loc[v]
            if realocacao:
                for p in locais:
                    if end_region == p: # viagens que terminam no mesmo lugar não são realocações
                        continue

                    # Só é possível a realocação se em algum momento ocorreu uma viagem com mesma origem e destino
                    if (end_region,p) in travel_possibilities or p is None:
                        viagens_realizadas[(v,p)] = m.addVar(vtype=GRB.BINARY, name='x_'+str(v)+'_'+str(p))
                        if p == None:
                            nome_viagem_sr.append('x_'+str(v)+'_'+str(p))
                        else:
                            nome_viagem_cr.append('x_'+str(v)+'_'+str(p))
            else:
                viagens_realizadas[(v,None)] = m.addVar(vtype=GRB.BINARY, name='x_'+str(v)+'_'+str(None))
                nome_viagem_sr.append('x_'+str(v)+'_'+str(None))

        # Variável que indica quantos veículos estão em cada estação em dado momento
        print('Variável que indica quantos veículos estão em cada estação em dado momento')
        veiculos_ociosos = {}
        nome_veiculos_o = []
        for l in locais:
            for t in time_instants:
                veiculos_ociosos[(l,t)] = m.addVar(vtype=GRB.INTEGER, name='e_'+str(l)+'_'+str(t.value))
                nome_veiculos_o.append('e_'+str(l)+'_'+str(t.value))


        # Restrição de fluxo
        print('Restrição de fluxo')
        viagens_r = viagens_realizadas.keys()

        for t in range(1,len(time_instants)): #para todo instante t
            viagens_t = data[data.Start_time == time_instants[t-1]].index.values #tempo anterior
            viagens_t2 = data[data.Start_time == time_instants[t]].index.values #atual
            for p in locais: #ponto de parada
                viagens_p = data[data['end_region'] == p].index.values #viagens de outros lugares para p
                viagens_p2 = data[data['start_region'] == p].index.values # viagens de um ponto específico para outros lugares
                m.addConstr(
                    veiculos_ociosos[p,time_instants[t-1]] + quicksum(viagens_realizadas[v,p] for v in viagens_p if (v,p) in viagens_r) == quicksum(viagens_realizadas[v,p]for v in viagens_p2 if (v,p) in viagens_r) + veiculos_ociosos[p,time_instants[t]],"fluxo_"+str(p)+"_"+str(time_instants[t].value)
                )


        # Restrição para indicar o número de veículos em cada estação
        print('Restrição para indicar o número de veículos em cada estação')
        m.addConstr(quicksum(veiculos_ociosos[p,time_instants[0]] for p in locais) == n_vehicles, name='n_veiculos')


        # Restrição para evitar veículos realizarem mais de uma viagem ao mesmo tempo
        print('Restrição para evitar veículos realizarem mais de uma viagem ao mesmo tempo')
        for v in viagem_id:
            m.addConstr(
                quicksum(viagens_realizadas[v,p] for p in locais if (v,p) in viagens_realizadas.keys())==1
            , name='quant_viagem_'+str(v))



        # função objetivo
        print('Função objetivo')

        viagens = data[['start_region', 'end_region']].values

        m.setObjective(quicksum(
            quicksum(

                calcula_custo_realocacao(viagens, i, custo_realocacao, custo_real, p, bonus) * viagens_realizadas[i,p]

                for p in locais if (i,p) in viagens_realizadas.keys())
            for i in viagem_id)
            - quicksum(
                quicksum(
                    custo_fixo*veiculos_ociosos[p,t]
                    for t in time_instants)
            for p in locais), GRB.MAXIMIZE)

        if not os.path.exists('data'):
            os.makedirs('data')

        m.write("data/realocacao_model.lp")
        print('\n')
    else:
        print("Lendo dados lp...")
        m = read(model_lp)


    print('Otimização..')
    m.Params.TimeLimit = 5400 # tempo limite de 1h e meia
    print(m.optimize())
    m.write("data/realocacao_solution.sol")

    t_final = datetime.now()

    duracao = t_final - t_inicial
    duracao = duracao.total_seconds()
    t_inicial = t_inicial.strftime("%d/%m/%Y %H:%M:%S")
    
    inicio_coleta = data.Start_time.min()
    viagens = data[['start_region', 'end_region', 'Start_time', 'End_time']].values
    grafo_labels = {}
    grafo_colors = {}
    color = []
    custo_ociosos = 0
    for nome in nome_veiculos_o:
        #grafo.add_node(tuple(nome.split('_')))
        custo_ociosos += m.getVarByName(nome).x * custo_fixo

    inicio_coleta = data.Start_time.min()
    ##calculando as variaveis
    lucro_viagens_sr = 0
    for nome in nome_viagem_sr:
        if m.getVarByName(nome).x == 1.0:
            _, viagem_id, _ = nome.split('_')
            viagem = viagens[int(viagem_id)]

            # formando label para o grafo
            inicio_segundos = (viagem[2] - inicio_coleta).total_seconds()
            fim_segundos = (viagem[3] - inicio_coleta).total_seconds()

            no_inicial = 'p(' + str(int(viagem[0])) + ')_t(' + str(int(inicio_segundos)) + ')'
            no_final = 'p(' + str(int(viagem[1])) + ')_t(' + str(int(fim_segundos)) + ')'

            grafo_labels[(viagem[0],viagem[2])] = no_inicial
            grafo_labels[(viagem[1],viagem[3])] = no_final
            grafo_colors[(viagem[0],viagem[2])] = 'b'
            grafo_colors[(viagem[1],viagem[3])] = 'b' 
            color.append('b')
            color.append('b')

            grafo.add_edge(no_inicial,no_final) # adicionando aresta ao grafo

            lucro_viagens_sr += m.getVarByName(nome).x * custo_real[(viagem[0],viagem[1])]
            
    custo_viagens_cr = 0 
    lucro_viagens_cr = 0   
    cont_realoc = 0    
    label={}
    i = 0  
    for nome in nome_viagem_cr:
        if m.getVarByName(nome).x == 1.0:
            cont_realoc+=1
            _, viagem_id, destino = nome.split('_')
            viagem = viagens[int(viagem_id)]

            lucro_viagens_cr += m.getVarByName(nome).x * custo_real[(viagem[0],viagem[1])] # custo da viagem
            
            inicio_segundos = (viagem[2] - inicio_coleta).total_seconds()
            fim_segundos = (viagem[3] - inicio_coleta).total_seconds()

            no_inicial = 'p(' + str(int(viagem[0])) + ')_t(' + str(int(inicio_segundos)) + ')'
            no_final = 'p(' + str(int(viagem[1])) + ')_t(' + str(int(fim_segundos)) + ')'

            grafo_labels[(viagem[0],viagem[2])] = no_inicial
            grafo_labels[(viagem[1],viagem[3])] = no_final
            grafo_colors[(viagem[0],viagem[2])] = 'b'
            grafo_colors[(viagem[1],viagem[3])] = 'b' 
            color.append('b')
            color.append('b')

            grafo.add_edge(no_inicial,no_final) # adicionando aresta ao grafo
            

            # aresta de realocação
            valor_realoc = bonus * m.getVarByName(nome).x * custo_realocacao[(viagem[1],float(destino))] # custo da realocacao
            minutos = int(custo_realocacao[(viagem[1],float(destino))]/0.41)
            tempo = timedelta(minutes = minutos) + viagem[3]
            realocacao_segundos = (tempo - inicio_coleta).total_seconds()

            no_inicial = 'p(' + str(int(viagem[1])) + ')_t(' + str(int(fim_segundos)) + ')'
            no_final = 'p(' + str(int(float(destino))) + ')_t(' + str(int(realocacao_segundos)) + ')'

            grafo.add_edge(no_inicial,no_final) # adicionando aresta ao grafo
            
            
            custo_viagens_cr += valor_realoc

    pos =graphviz_layout(grafo, prog='dot')

    nx.draw(grafo, pos, with_labels=True)
    plt.savefig('data/grafo.png')

    nx.draw_networkx_labels(grafo, pos, label, font_size=16)

    nx.draw(grafo, with_labels=True, font_weight='bold')
    write_dot(grafo, 'grafo.dot')
#    plt.show()
    plt.savefig("grafo.png")
    
    if not os.path.exists('data/model_data.csv'):
        with open('data/model_data.csv', mode='a') as table:
            table_writer = csv.writer(table, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            table_writer.writerow(['inicio','arquivo','espaco_tempo','intervalo_tempo','n_veiculos',
                                   'bonus','duracao(seg)','tamanho_amostra','realocacao','valor_objetivo',
                                   'Gap', 'lucro_viagens_sr', 'lucro_viagens_cr', 'custo_viagens_cr', 'quant_realoc', 'custo_ociosos'])

    with open('data/model_data.csv', mode='a') as table:
        table_writer = csv.writer(table, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        table_writer.writerow([t_inicial, path, time_space, time_interval, n_vehicles,
                               bonus, duracao, len(data), realocacao, m.ObjVal, 
                               m.MIPGap, lucro_viagens_sr, lucro_viagens_cr, custo_viagens_cr, cont_realoc, custo_ociosos ])


if __name__ == "__main__":
    main(sys.argv[1:])