from pandas import Series
from statsmodels.tsa.stattools import adfuller

def diffs_estacionariedade(valores_serie):

    serie = Series(valores_serie, index=range(1, valores_serie.size+1))

    H0_raiz_unitaria = True
    d = 0
    limite_diferencas = 3
    nivel_significancia = 0.05

    while H0_raiz_unitaria:

        resultado = adfuller(serie.values)
        valor_p = resultado[1]

        if valor_p < nivel_significancia:
            H0_raiz_unitaria = False
        else:
            if d == limite_diferencas:
                break
            else:

                serie -= serie.shift()
                serie.dropna(inplace=True)
                d += 1

    return d, serie