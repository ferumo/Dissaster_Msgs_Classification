# Dissaster Tweets Multilabel Model on WebApp
Multi label classification web app for dissaster messages <br>
Date: 05-Apr-2023

<b>Medium Blog Link:</b> [Have you ever wondered? Analysis of the Airbnb rentals in Mexico City](https://medium.com/@frubi17/have-you-ever-wondered-analysis-of-the-airbnb-rentals-in-mexico-city-a337e33f0ed4)

## Objective
The objective of this project is to answer 4 questions regarding the Airbnb rentals of Mexico City.<br>
The first 2 questions are as a User and the final 2 questions are for a possible Host or rental owner.

### Questions:
1. How many days before a main holiday should I reserve to still have several options?
2. If I want to try to avoid crowded places or traffic jams on the highway, which day of the week should I travel?
3. What neighborhood is the best for having an Airbnb in Mexico City?
4. Does the size matter? What are the most important features to have a high occupancy?

### Libraries used
matplotlib==3.5.1 <br>
numpy==1.22.1 <br>
pandas==1.4.2 <br>
plotly==5.10.0 <br>
scikit-learn==1.1.3 <br>
seaborn==0.11.2 <br>
tensorflow-intel==2.11.0 <br>
folium==0.5.0 <br>

### Files Description
<b>Airbnb_Analysis_CDMX.ipynb:</b> Notebook containing the complete project of the data analysis and model classification <br>
<b>cdmx_map.zip:</b> Zipped file containing the HTML of the Interactive Map with all the rental propierties color coded by occupancy classification

<b>Datasets</b> ([Link](http://insideairbnb.com/get-the-data/))
- Mexico City 26-Mar-2022 (calendar.csv.gz, listings.csv.gz)
- Mexico City 21-Jun-2022 (calendar.csv.gz, listings.csv.gz)
- Mexico City 22-Sept-2022 (calendar.csv.gz, listings.csv.gz)
- Mexico City 29-Dec-2022 (calendar.csv.gz, listings.csv.gz)

### Results Summary
The main findings for this projects are, to plan your vacations at least 4 months earlier and if it's not possible at least try to avoid peak days like Friday or Saturday to travel.
Also when using Airbnb always check the reviews in particular the 'Host response rate' and as a Host always try to motivate the user to give you back good reviews they'll help you get higher occupancy to your rentals.

### Acknowledgements
Mexico City Airbnb Datasets credit to [Inside Airbnb](http://insideairbnb.com/get-the-data/)<br>
Ashish, K. Occupancy Rate Article from [WallStreetMojo](https://www.wallstreetmojo.com/occupancy-rate/)
