RealocacaoVeiculos

Grupo: Bryan Barbosa, Gabriele Iwashima e Victor Aquiles Alencar

Para rodar pelo terminal utilize a seguinte estrutura:

python modelo.py <caminho> <Espaço de tempo> <Intervalo de tempo>

Onde:
Caminho: 
	caminho dos dados a serem lidos

Espaço de tempo: 
	indica a granulariedade do tempo use m->minutos, h->hora

Intervalo de tempo: 
	indica o intervalo de tempo dos dados no formato: n-p,
	p - indica a granulariedade: d->dias, w->semanas e m->meses
	n - indica a quantidade de intervalos de tempo
	Ex: 5-w, irá ler dados de 5 semanas

Exemplo: python modelo.py ../data/evo_travels.csv h 1-d
