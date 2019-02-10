from pandas import Series
from pandas import DataFrame

from statsmodels.tsa.stattools import adfuller
from itertools import product
from random import uniform
from math import inf
import numpy as np
from scipy.stats import chi2

class Modelo:
    p = 0
    q = 0
    pesos_autoregressivos = []
    pesos_medias_moveis = []
    ruidos_estimados = []
    Q_valor_portmanteau = 0
    p_valor_portmanteau = 0

def diffs_estacionariedade(serie_temporal):

    H0_raiz_unitaria = True
    d = 0
    nivel_significancia = 0.1
    limite_diferencas = 3

    while H0_raiz_unitaria:

        resultado = adfuller(serie_temporal.values)
        valor_p = resultado[1]

        if valor_p < nivel_significancia:
            H0_raiz_unitaria = False
        else:
            if d == limite_diferencas:
                break
            else:

                serie_temporal -= serie_temporal.shift()
                serie_temporal.dropna(inplace=True)
                d += 1

    return d, serie_temporal

def amostragem_pesos(qtd_pesos, limite_inferior=-1, limite_superior=1):

    pesos = []

    for num_peso in range(qtd_pesos):

        peso_gerado = uniform(limite_inferior,limite_superior)
        pesos.append(peso_gerado)

    return pesos

def adequacao_polinomio(pesos):

    if len(pesos) == 0:
        return True

    pesos = np.flip(pesos)
    pesos *= -1
    pesos = np.append(pesos, 1)

    raizes_polinomio = np.roots(pesos)
    raizes_polinomio = abs(raizes_polinomio)

    adequacao = raizes_polinomio.all() > 1

    return adequacao

def soma_quadrados(serie, d, pesos_AR, pesos_MA):

    if d == 0:

        media = serie.mean()
        serie -= media

    estimacao_ruidos = DataFrame(serie, columns = ['Wt'])
    estimacao_ruidos.index = range(1, serie.size + 1)
    estimacao_ruidos['At'] = 0

    for indice_ruido in range(len(pesos_AR)+1, serie.size + 1):

        valor_estimado = estimacao_ruidos['Wt'][indice_ruido]

        for indice_AR in range(len(pesos_AR)):

            valor_estimado -= pesos_AR[indice_ruido]*estimacao_ruidos['Wt'][indice_ruido-(indice_AR+1)]

        for indice_MA in range(len(pesos_MA)):

            posicao_valor_passado = indice_ruido - (indice_MA + 1)
            if posicao_valor_passado > 0:
                valor_estimado += pesos_MA[indice_MA]*estimacao_ruidos['At'][posicao_valor_passado]

        estimacao_ruidos['At'][indice_ruido] = valor_estimado


    resultado_soma = (estimacao_ruidos['At'] ** 2).sum()

    intervalo_ruidos_estimados = range(len(pesos_AR)+1, serie.size+1)
    ruidos_estimados = estimacao_ruidos['At'].loc[intervalo_ruidos_estimados]
    ruidos_estimados.index = range(1, ruidos_estimados.size+1)

    return resultado_soma, ruidos_estimados

def estimacao_parametros(modelo, serie, qtd_diferencas):

    orcamento_busca = 10
    minimo_alcancado = inf

    for num_rodada in range(orcamento_busca):

        pesos_AR = amostragem_pesos(modelo.p)
        pesos_MA = amostragem_pesos(modelo.q)

        if (not adequacao_polinomio(pesos_AR)) or (not adequacao_polinomio(pesos_MA)):
            pass

        resultado_soma, ruidos_estimados = soma_quadrados(serie, qtd_diferencas, pesos_AR, pesos_MA)

        if resultado_soma < minimo_alcancado:

            minimo_alcancado = resultado_soma

            modelo.pesos_autoregressivos = pesos_AR
            modelo.pesos_medias_moveis = pesos_MA
            modelo.ruidos_estimados = ruidos_estimados

    return modelo

def calcula_autocovariancia(lag, serie):

    tamanho_amostra = serie.size
    media_amostral = serie.mean()

    estimativa_autocovariancia = 0

    for t in range(1, tamanho_amostra-lag+1):

        estimativa_autocovariancia += (serie[t] - media_amostral) * (serie[t+lag] - media_amostral)

    estimativa_autocovariancia /= tamanho_amostra

    return estimativa_autocovariancia

def calcula_autocorrelacao(lag, serie):

    estimativa_gamma_k = calcula_autocovariancia(lag, serie)
    estimativa_gamma_0 = calcula_autocovariancia(0, serie)

    estimativa_autocorrelacao = estimativa_gamma_k / estimativa_gamma_0

    return estimativa_autocorrelacao

def portmanteau_teste(modelo, K=25):

    n = modelo.ruidos_estimados.size

    soma_autocorrelacoes = 0
    for k in range(1, K+1):

        autocorrelacao = calcula_autocorrelacao(k, modelo.ruidos_estimados)
        denominador = n - k
        soma_autocorrelacoes += (autocorrelacao**2)/denominador

    estatistica_Q = n*(n+2) * soma_autocorrelacoes

    graus_liberdade = K - modelo.p - modelo.q
    valor_p = 1 - chi2.cdf(estatistica_Q, graus_liberdade)

    return estatistica_Q, valor_p


caminho = '..\Datasets\series_a.csv'
serie_temporal = Series.from_csv(caminho, index_col=None)

d, serie_diferenciada = diffs_estacionariedade(serie_temporal)

modelos_ajustados = []

for quantidades_pesos in product(range(5), repeat=2):

    p = quantidades_pesos[0]
    q = quantidades_pesos[1]

    modelo = Modelo()
    modelo.p = p
    modelo.d = d
    modelo.q = q

    modelo = estimacao_parametros(modelo, serie_diferenciada, d)
    modelos_ajustados.append(modelo)

maximo_p = 0
for modelo in modelos_ajustados:

    estatistica_Q, valor_p = portmanteau_teste(modelo)

    modelo.Q_valor_portmanteau = estatistica_Q
    modelo.p_valor_portmanteau = valor_p

    if valor_p > maximo_p:

        maximo_p = valor_p
        modelo_selecionado = modelo


# Usar o modelo selecionado para exibir suas informações e
# a comparação entre a serie original e os pontos estimados por esse modelo
