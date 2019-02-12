from math import inf
from random import uniform
import numpy as np
from pandas import DataFrame

def amostragem_pesos(qtd_pesos, limite_inferior=-1, limite_superior=1):

    pesos = []

    for num_peso in range(qtd_pesos):

        peso_gerado = uniform(limite_inferior,limite_superior)
        peso_gerado = round(peso_gerado, 2)
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
    estimacao_ruidos['At'] = 0.0

    for indice_ruido in range(len(pesos_AR)+1, serie.size + 1):

        valor_estimado = estimacao_ruidos['Wt'][indice_ruido]

        for indice_AR in range(len(pesos_AR)):

            valor_estimado -= pesos_AR[indice_AR]*estimacao_ruidos['Wt'][indice_ruido-(indice_AR+1)]

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

    orcamento_busca = 100
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
