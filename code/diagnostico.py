from scipy.stats import chi2
from pandas import Series

class Resultado_portmanteau:

    estatistica_Q = 0
    valor_p = 0
    autocorrelacoes = []

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

    autocorrelacoes_ruidos = []
    soma_autocorrelacoes = 0

    for k in range(1, K+1):

        autocorrelacao = calcula_autocorrelacao(k, modelo.ruidos_estimados)
        autocorrelacoes_ruidos.append(autocorrelacao)
        denominador = n - k
        soma_autocorrelacoes += (autocorrelacao**2)/denominador

    estatistica_Q = n*(n+2) * soma_autocorrelacoes

    graus_liberdade = K - modelo.p - modelo.q
    valor_p = 1 - chi2.cdf(estatistica_Q, graus_liberdade)

    resultado = Resultado_portmanteau()
    resultado.estatistica_Q = estatistica_Q
    resultado.valor_p = valor_p
    resultado.autocorrelacoes = Series(autocorrelacoes_ruidos, index=range(1, K+1))

    return resultado
