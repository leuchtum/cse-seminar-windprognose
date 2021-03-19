# 01_literature_review

Dieses Dokument stellt sieben Paper vor, die in Bezug auf Windprognose interessant sind.

### Absatz 1

> **ACHTUNG**: [**SIEHE DOKUMENT**](./Orpia_2014_Time%20Series%20Analysis%20using%20Vector%20Auto%20Regressive%20(VAR)%20Model%20of%20Wind%20Speeds.pdf) = Absatz 2

* Bei Windprognosen ist es möglich, örtliche und zeitliche Abhängigkeiten mit zu berücksichtigen
* Wenn nur Daten von einem Ort vorliegen: -> Model via:
  + AR (Autoregressive Modelle)
  + VAR (Vektorautoregressive Modelle)
  + ARIMA (ein Volatilitätsmodell)

### **Absatz 2** - Orpia, C., Mapa, D. S., & Orpia, J. (2014). Time Series Analysis using Vector Auto Regressive (VAR) Model of Wind Speeds in Bangui Bay and Selected Weather Variables in Laoag City, Philippines.

* Vorhandenes Paper: Prognose von Windgeschw. über tägliche Daten
* Modell: VAR-Modell unter Verwendung täglicher Zeitreihendaten
* Genutzte Variablen:
  + Windgeschw.
  + Luftfeuchtigkeit
  + Temp.
  + Luftdruck
* Diese Faktoren beeinflussen die Windturbine
* Diese Faktoren sind stationär (nach dem Augmented Dickey-Fuller-Test)
* Ergebnisse von vorhandenen Paper:
  + Windgeschw. erklärbar mit AR(1)
  + Steigen Temperatur, Luftfeuchtigkeit oder Luftdruck, sinkt Windgeschw.
  + Gezeigt durch Impulse Response Funktion (Impulsantwortfunktion)

### **Absatz 3** - Dowell, J. (2015). Spatio-temporal prediction of wind fields(Doctoral dissertation, University of Strathclyde).

* Vorhandenes Paper: Prognose von Windgeschw. an mehreren Standorten für Windkraftanlagen unter Verwendung räumlich-zeitlicher Informationen
* Modell: Sparse-VAR (sVAR)
* Horizont: 5min Vorhersage der Windleistung in 22 Windparks innerhalb eines Jahres
* Es werden nur Lastdaten der Turbinen zur Prognose genutzt
* Vergleich mit konventionallen VAR und AR
* Eröffnet Verbesserungsmöglichkeiten (Zuverlässigkeit etc.)

### **Absatz 4** - Ewing, B. T., Kruse, J. B., Schroeder, J. L., & Smith, D. A. (2007). Time series analysis of wind speed using VAR and the generalized impulse response technique. Journal of wind engineering and industrial aerodynamics, 95(3), 209-219.

* Vorhandenes Paper: Interessant, analysiert Windgeschwindigkeiten in verschiedenen Höhen in Abhängigkeit der Höhe.
* Nutzt VAR Model (VAR(3) mit mehreren Gleichungen) und Innovationsrechnungsmethode/Impulsantwortanalyse

### **Absatz 5** - He, M., Vittal, V., & Zhang, J. (2015, July). A sparsified vector autoregressive model for short-term wind farm power forecasting. In Power & Energy Society General Meeting, 2015 IEEE (pp. 1-5). IEEE.

* Vorhandenes Paper: Kurzzeitige Leistungsprognose (Windprognose) innerhalb eines Windparks über die aktuelle Leistung der anderen Turbinen (=räumlich-zeitliche Korrelation)
* Nutzt VAR Model
* Umgehen Gefahr von Overfitting über eine clevere abgespeckte AR Koeffizientenmatrix, die Windrichtung, Geschw. und Layout des Windparks (?Leistung?) berücksichtigt.
* Multivariate Windleistungsdaten auf Turbinenebene haben hohes Potential zur Verbesserung von Lastprognose/Modellierung verglichen mit Verwendung von nur aggregierten Windleistungsdaten

### **Absatz 6** - Koivisto, M., Seppänen, J., Mellin, I., Ekström, J., Millar, J., Mammarella, I., ... & Lehtonen, M. (2016). Wind speed modeling using a vector autoregressive process with a time-dependent intercept term. International Journal of Electrical Power & Energy Systems, 77, 91-99.

* Vorhandenes Paper: Windprognose an verschiedenen Orten, berücksichtigt zeitliche und räumliche Abhängigkeitsstrukturen im Detail
* Nutzt spezielles VARTA Model (time-dependent intercept/ -> ?"zeitabhängige Interpolation"?)
* Lang und Kurzzeitprognose
* Daten von 21 Standorten in Finnland
* Guter Fit durch VARTA
* VARTAX Nutzt zusätzlich noch Monte Carlo Simulation, um sehr schwache und sehr starke Winde mit simulieren zu können.

### **Absatz 7** - Erdem, E., & Shi, J. (2011). ARMA based approaches for forecasting the tuple of wind speed and direction. Applied Energy, 88(4), 1405-1414.

> **ACHTUNG**: Könnte für unsere Seminararbeit nützlich sein (Ansatz ähnlich)

* Vier Ansätze mit ARMA (autoregressive moving average)
* Vorgehen (Nicht so leicht aus der Zusammenfassung zu verstehen, am besten originales Paper lesen):
  1. Geschw.-Vektor wird in Längen/Breitengrad Vektoren zerteilt. 
  2. Beide werden durch ein jeweiliges ARMA Model repräsentiert. Traditionelles ARMA für Geschw. und ein ?verlinktes? Model für Richtung.
  3. VAR Model um die Richtungs/Geschw. Tupel zvorherzusagen.
  4. eingeschränkte Version des VAR-Ansatzes zur Vorhersage des Tupels von Windattributen (Richtung/Geschw.)
* 1h Horizont
* Bewertet durch MAE (Mittlerer absoluter Fehler)
* Ergebnisse: Ihr Komponenten-Model ist besser als ein traditionell ?verlinktes? ARMA Model. Vorschlag, dass Verwendung einer eingeschränkten Version der VAR-Modelle ein Schlüssel für einfachere Modelle sein könnte.

### **Absatz 8** - Torres, J. L., Garcia, A., De Blas, M., & De Francisco, A. (2005). Forecast of hourly average wind speed with ARMA models in Navarre (Spain). Solar Energy, 79(1), 65-77.

* Vorhandenes Paper: Prognose des stündlichen Mittelwertes mit bis zu 10h Horizont
* Modell: ARMA (Autoregressiver gleitender Durchschnittsprozess) und Persistenzmodelle
* Autoren haben Zeitreihen transformiert und standardisiert wegen nicht-gaussschen Natur der stündlichen Windgeschwindigkeitsverteilung und der nicht-stationären Natur ihrer täglichen Entwicklung.
* Wegen Saisonalen Effekten: 1 Model pr Monat
* Bis zu 20% besser als im Vergleich zu ?vorhandenen (nochmal nachlesen!!!)? Modellen
* Weitere Erkenntnisse:
  * Wenn Geschw im Bereich von 4-11m/s: Geringer Fehler, wenig abhängig von der Prognoselänge
  * Die größten Fehler treten bei extremen Windgeschwindigkeiten auf (irrelevant, da Turbinen eh ab einer gewissen Stärke abschalten.)