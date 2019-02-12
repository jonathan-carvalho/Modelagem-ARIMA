from identificacao import *
from estimacao import *
from diagnostico import *
from graficos_resultado import *

import sys
from pandas import Series
from itertools import product

class Modelo:
    p = 0
    d = 0
    q = 0
    pesos_autoregressivos = []
    pesos_medias_moveis = []
    ruidos_estimados = []
    autocorrelacoes_ruidos = []
    Q_valor_portmanteau = 0
    p_valor_portmanteau = 0

arquivo_serie = sys.argv[1]
caminho = '..\Datasets\\' + arquivo_serie + '.csv'
serie_temporal = Series.from_csv(caminho, index_col=None)
serie_temporal.index = range(1, serie_temporal.size+1)

d, serie_diferenciada = diffs_estacionariedade(serie_temporal.values)

print('Estimação dos parâmetros:')
modelos_ajustados = []

for quantidades_pesos in product(range(5), repeat=2):

    p = quantidades_pesos[0]
    q = quantidades_pesos[1]

    print('Modelo ARIMA(' + str(p) + ',' + str(d) + ',' + str(q) + ')')

    modelo = Modelo()
    modelo.p = p
    modelo.d = d
    modelo.q = q

    modelo = estimacao_parametros(modelo, serie_diferenciada, d)
    modelos_ajustados.append(modelo)

print('\nDiagnosticando os modelos ...')

maximo_p = 0
for modelo in modelos_ajustados:

    resultado_portmanteau = portmanteau_teste(modelo)

    modelo.Q_valor_portmanteau = round(resultado_portmanteau.estatistica_Q, 2)
    modelo.p_valor_portmanteau = round(resultado_portmanteau.valor_p, 2)
    modelo.autocorrelacoes_ruidos = resultado_portmanteau.autocorrelacoes

    if modelo.p_valor_portmanteau > maximo_p:

        maximo_p = modelo.p_valor_portmanteau
        modelo_selecionado = modelo


print('\nModelo Selecionado: ARIMA(' + str(modelo_selecionado.p) +
                                ',' + str(modelo_selecionado.d) +
                                ',' + str(modelo_selecionado.q) + ')')
print('Pesos AR:')
print(modelo_selecionado.pesos_autoregressivos)
print('Pesos MA:')
print(modelo_selecionado.pesos_medias_moveis)

print('Estatística Q: ' + str(modelo_selecionado.Q_valor_portmanteau))
print('Valor p: ' + str(modelo_selecionado.p_valor_portmanteau))

exibir_fac_ruido(modelo_selecionado.autocorrelacoes_ruidos)
comparacao_esperado_obtido(serie_temporal, modelo_selecionado)
