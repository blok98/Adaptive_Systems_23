# Adaptive_Systems_23

In deze repo staat de code en de documentatie van opdracht 1.2 van Adaptive Systems 22-23. Voor elke section is een eigen branch aangemaakt.

Branch:
- <b>Model-based-agent:</b> section A&B
- <b>Utility-based-agent:</b> section C
- <b>Stochasticische-omgeving:</b> extra
- <b>Perceptual-Control-Theory:</b> reconstructie code op basis van PCT structuur

Elke functie is voorzien van docstrings waarin wordt beschreven wat de functies en argumenten doen.

Bij elke sectie, behalve bij A & B, wordt een visualisatie gemaakt met pygame waarin het gedrag van de agent live gevolgd kan worden.
In main.py kan aan het einde van de while loop de timer worden aangepast om de pauzes te verlagen en het model sneller te laten runnen.
Ook kan de exploration rate aangepast worden in de iterate functie die ook aan het einde van de while loop staat.

Het verslag van AS1.2 geeft verder toelichting over de theorie, structuur van de code en de resultaten.

### visualisatie
Lichtblauw is de begin state, dat nu op (0,0) staat aangezien we alle states afgaan, donkerblauw is de huidige positie van de agent en de kleuren van rood naar groen laten zien hoe hoog de utility en de reward is. De reward wordt getoond wanneer de witte button wordt aangeklikt. De snelheid van de agent wordt bepaald door de sleep functie aan het einde van de while loop.

<i> Dependencies:
  - pygame</i>
