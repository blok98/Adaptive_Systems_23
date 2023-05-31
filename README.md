# Adaptive_Systems_23

In deze repo staat de code en de documentatie van opdracht 1.2 & 2.2 van Adaptive Systems 22-23. Voor elke section is een eigen branch aangemaakt.

Branch:
- <b>Model-based-agent:</b> 1.2 section A&B
- <b>Utility-based-agent:</b> 1.2 section C
- <b>Stochasticische-omgeving:</b> 1.2 extra
- <b>Perceptual-Control-Theory:</b> 1.2 reconstructie code op basis van PCT structuur
- <b>Model-free-prediction-&-control: </b> 2.2 section C

Elke functie is voorzien van docstrings waarin wordt beschreven wat de functies en argumenten doen.

<b>1.2</b>
Bij elke sectie, behalve bij A & B van 1.2, wordt een visualisatie gemaakt met pygame waarin het gedrag van de agent live gevolgd kan worden.
In main.py kan aan het einde van de while loop de timer worden aangepast om de pauzes te verlagen en het model sneller te laten runnen.
Ook kan de epsilon aangepast worden in de iterate functie die ook aan het einde van de while loop staat. Dit is meer een voorbereiding voor opdrachten van 2.

In de branch Utility-based-agent is de delta toegevoegd en wordt de policy gevisualiseerd en gereturned. Vanwege tijdsgebrek is dit niet toegepast voor de andere branches.
Let op dat het runnen van de code met de delta de window blijft updaten maar de agent geen acties meer neemt. Dit is express gedaan om het resultaat in de window te blijven tonen.
De window kan simpelweg gesloten worden zodat de utilities en de policy in de terminal worden geprint.

Het verslag van AS1.2 geeft verder toelichting over de theorie, structuur van de code en de resultaten.

<b>2.2</b>
Om de simulatie van 2.2 te runnen kan in main.py - __main__ de run() functie en de visualize_maze() functie aangeroepen worden. Visualize_maze() laat stap voor stap zien hoe de qvalues worden geupdatet in een pygame visualisatie. De run() functie is bedoeld om grote hoeveelheden episodes te runnen en laat alleen de laatste stappen binnen een visualisatie zien. De parameters voor beide functies zijn hetzelfde en omvatten de agent, de maze, step_time = 0.3, discount_factor, learning_rate ,epsilon, en n_episodes. De step_time geeft aan hoe snel de visualisatie runt, maar heeft geen invloed op de runtime bij run(). 

De huidige settings die gebruikt zijn:
- discount_factor: 1
- learning_rate: 0.2
- epsilon: 0.2
- n_episodes: 1000000

<b> visualisatie </b><br>
Lichtblauw is de begin state, dat nu op (0,0) staat aangezien we alle states afgaan, donkerblauw is de huidige positie van de agent en de kleuren van rood naar groen laten zien hoe hoog de utility en de reward is. De reward wordt getoond wanneer de witte button wordt aangeklikt. De snelheid van de agent wordt bepaald door de sleep functie aan het einde van de while loop.

<i> Dependencies:
  - pygame</i>
