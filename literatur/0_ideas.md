- Im Paper wird 1-360 Grad + Windgeschw. genutzt, wir sollten Wind lieber in zwei Vektoren aufteilen. 
- Prognose von Vx und Vy. Damit können wir Windgeschwindigkeit und Windrichtung vorhersagen.
- Wir sollten nicht nur die aktuellen Daten, sondern auch z.b. die letzten 24h mit hinzuziehen
- Tag und Saison als Sinus codieren und dem NN mitgeben
- Jedes Subnetz wird mit jedem Kriterium einmal trainiert. Daraus ergeben sich also 102 * 4 Netze. Nun hat er für jedes Kriterium den mittleren MSE berechnet. Das Kriterium mit dem geringsten mittleren MSE ist das "Gewinnerkriterium". Unserer Ansicht nach wäre es aber besser nicht das Kriterium mit dem geringsten mittleren Fehler, sondern die Subnetze mit dem jeweiligen geringsten Fehler zu nutzen und aus diesen das Ensemble-Netz zu erstellen.