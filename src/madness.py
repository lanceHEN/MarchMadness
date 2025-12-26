from bracket import BaseGame, Team, UpperGame
from model import LogisticRegression, MLP
from optimize import find_max_bracket

# using seeding predictions from ESPN bracketology as of 12/24/24

# init teams

# south
south111 = Team("Auburn", 1)
south112 = Team("Alabama St.", 16)

south121 = Team("Louisville", 8)
south122 = Team("Creighton", 9)

south131 = Team("Michigan", 5)
south132 = Team("UC San Diego", 12)

south141 = Team("Texas A&M", 4)
south142 = Team("Yale", 13)

south151 = Team("Mississippi", 6)
south152 = Team("Xavier", 11)

south161 = Team("Iowa St.", 3)
south162 = Team("Lipscomb", 14)

south171 = Team("Marquette", 7)
south172 = Team("New Mexico", 10)

south181 = Team("Michigan St.", 2)
south182 = Team("Bryant", 15)

# west
west111 = Team("Florida", 1)
west112 = Team("Norfolk St.", 16)

west121 = Team("Connecticut", 8)
west122 = Team("Oklahoma", 9)

west131 = Team("Memphis", 5)
west132 = Team("Colorado St.", 12)

west141 = Team("Maryland", 4)
west142 = Team("Grand Canyon", 13)

west151 = Team("Missouri", 6)
west152 = Team("Drake", 11)

west161 = Team("Texas Tech", 3)
west162 = Team("UNC Wilmington", 14)

west171 = Team("Kansas", 7)
west172 = Team("Arkansas", 10)

west181 = Team("St. John's", 2)
west182 = Team("Nebraska Omaha", 15)

#midwest
midwest111 = Team("Duke", 1)
midwest112 = Team("American", 16)

midwest121 = Team("Mississippi St.", 8)
midwest122 = Team("Baylor", 9)

midwest131 = Team("Oregon", 5)
midwest132 = Team("Liberty", 12)

midwest141 = Team("Arizona", 4)
midwest142 = Team("Akron", 13)

midwest151 = Team("BYU", 6)
midwest152 = Team("VCU", 11)

midwest161 = Team("Wisconsin", 3)
midwest162 = Team("Montana", 14)

midwest171 = Team("Saint Mary's", 7)
midwest172 = Team("Vanderbilt", 10)

midwest181 = Team("Alabama", 2)
midwest182 = Team("Robert Morris", 15)

#east

east111 = Team("Houston", 1)
east112 = Team("SIU Edwardsville", 16)

east121 = Team("Gonzaga", 8)
east122 = Team("Georgia", 9)

east131 = Team("Clemson", 5)
east132 = Team("McNeese St.", 12)

east141 = Team("Purdue", 4)
east142 = Team("High Point", 13)

east151 = Team("Illinois", 6)
east152 = Team("North Carolina", 11)

east161 = Team("Kentucky", 3)
east162 = Team("Troy", 14)

east171 = Team("UCLA", 7)
east172 = Team("Utah St.", 10)

east181 = Team("Tennessee", 2)
east182 = Team("Wofford", 15)

# init model and train
#model = MLP(2025, n_hidden=2, lr=0.00003, width_hidden=32, dropout=0)
#model.train(epochs=300, lambda_=5)
model = LogisticRegression(2025, lr=0.0001)
model.train()
print(model.predict("Auburn", "Oklahoma", "N"))
print("Accuracy for 2024 season:",model.accuracy(year=2024))

# init games
# south
south11 = BaseGame(model, 1, south111, south112)
south12 = BaseGame(model, 1, south121, south122)

south13 = BaseGame(model, 1, south131, south132)
south14 = BaseGame(model, 1, south141, south142)

south15 = BaseGame(model, 1, south151, south152)
south16 = BaseGame(model, 1, south161, south162)

south17 = BaseGame(model, 1, south171, south172)
south18 = BaseGame(model, 1, south181, south182)

# west
west11 = BaseGame(model, 1, west111, west112)
west12 = BaseGame(model, 1, west121, west122)

west13 = BaseGame(model, 1, west131, west132)
west14 = BaseGame(model, 1, west141, west142)

west15 = BaseGame(model, 1, west151, west152)
west16 = BaseGame(model, 1, west161, west162)

west17 = BaseGame(model, 1, west171, west172)
west18 = BaseGame(model, 1, west181, west182)

#midwest
midwest11 = BaseGame(model, 1, midwest111, midwest112)
midwest12 = BaseGame(model, 1, midwest121, midwest122)

midwest13 = BaseGame(model, 1, midwest131, midwest132)
midwest14 = BaseGame(model, 1, midwest141, midwest142)

midwest15 = BaseGame(model, 1, midwest151, midwest152)
midwest16 = BaseGame(model, 1, midwest161, midwest162)

midwest17 = BaseGame(model, 1, midwest171, midwest172)
midwest18 = BaseGame(model, 1, midwest181, midwest182)

# east
east11 = BaseGame(model, 1, east111, east112)
east12 = BaseGame(model, 1, east121, east122)

east13 = BaseGame(model, 1, east131, east132)
east14 = BaseGame(model, 1, east141, east142)

east15 = BaseGame(model, 1, east151, east152)
east16 = BaseGame(model, 1, east161, east162)

east17 = BaseGame(model, 1, east171, east172)
east18 = BaseGame(model, 1, east181, east182)

#round 2 games
# south
south21 = UpperGame(model, 2, south11, south12)
south22 = UpperGame(model, 2, south13, south14)

south23 = UpperGame(model, 2, south15, south16)
south24 = UpperGame(model, 2, south17, south18)

# west
west21 = UpperGame(model, 2, west11, west12)
west22 = UpperGame(model, 2, west13, west14)

west23 = UpperGame(model, 2, west15, west16)
west24 = UpperGame(model, 2, west17, west18)

# midwest
midwest21 = UpperGame(model, 2, midwest11, midwest12)
midwest22 = UpperGame(model, 2, midwest13, midwest14)

midwest23 = UpperGame(model, 2, midwest15, midwest16)
midwest24 = UpperGame(model, 2, midwest17, midwest18)

# east
east21 = UpperGame(model, 2, east11, east12)
east22 = UpperGame(model, 2, east13, east14)

east23 = UpperGame(model, 2, east15, east16)
east24 = UpperGame(model, 2, east17, east18)

# round 3 games
# south
south31 = UpperGame(model, 3, south21, south22)
south32 = UpperGame(model, 3, south23, south24)

#west
west31 = UpperGame(model, 3, west21, west22)
west32 = UpperGame(model, 3, west23, west24)

# midwest
midwest31 = UpperGame(model, 3, midwest21, midwest22)
midwest32 = UpperGame(model, 3, midwest23, midwest24)

# east
east31 = UpperGame(model, 3, east21, east22)
east32 = UpperGame(model, 3, east23, east24)

# round 4 games
# south
south41 = UpperGame(model, 4, south31, south32)

# west
west41 = UpperGame(model, 4, west31, west32)

# midwest
midwest41 = UpperGame(model, 4, midwest31, midwest32)

# east
east41 = UpperGame(model, 4, east31, east32)

# round 5 games
left = UpperGame(model, 5, south41, west41)
right = UpperGame(model, 5, midwest41, east41)

# round 6 game
championship = UpperGame(model, 6, left, right)

opt_bracket, total = find_max_bracket(championship)
for round in range(1, 7):
    print(f"Round {round} winners: {[t.get_name for t in opt_bracket[round]]}")
    
print(f"Total expected points: {total}")