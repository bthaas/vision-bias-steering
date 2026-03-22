# Qualitative Generation Examples

Each template shown at λ=−100 (steer toward descriptive), λ=0 (no steering), λ=+100 (steer toward spatial).
Greedy decoding shown first, then beam (width=4).

## `image_shows` — "The image shows ___"

### λ=−100 (→descriptive)

**Greedy:**

- *Caption*: A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall
  *Output*: `The image shows very very very very very very very very very very very very very very very very very very very very`
- *Caption*: A bright orange and yellow maple tree stands beside a small dark pond, its round
  *Output*: `The image shows very very very very very very very very very very very very very very very very very very very very`
- *Caption*: A wide red double-decker bus is parked beside a row of tall narrow buildings, it
  *Output*: `The image shows , however, very very very very very very very very very very very very very very very very very`
- *Caption*: A tall iron gate stands at the far end of a narrow stone path, with a wooden ben
  *Output*: `The image shows very very very very very very very very very very very very very very very very very very very very`
- *Caption*: A round wooden table sits at the center of a small kitchen, with copper pots han
  *Output*: `The image shows , however, very very very very very very very very very very very very very very very very very`

**Beam (width=4):**

- *Caption*: A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall
  *Output*: `The image shows , however, a very very very very very very very very very very very very very very very very`
- *Caption*: A bright orange and yellow maple tree stands beside a small dark pond, its round
  *Output*: `The image shows , however, very very very very very very very very very very very very very very very very very`
- *Caption*: A wide red double-decker bus is parked beside a row of tall narrow buildings, it
  *Output*: `The image shows , however, very very very very very very very very very very very very very very very very very`
- *Caption*: A tall iron gate stands at the far end of a narrow stone path, with a wooden ben
  *Output*: `The image shows , however, a very very very very very very very very very very very very very very very very`
- *Caption*: A round wooden table sits at the center of a small kitchen, with copper pots han
  *Output*: `The image shows , however, very very very very very very very very very very very very very very very very very`

### λ=0 (unsteered)

**Greedy:**

- *Caption*: A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall
  *Output*: `The image shows a small, narrow, and narrow road that leads to a small town, and the road is very`
- *Caption*: A bright orange and yellow maple tree stands beside a small dark pond, its round
  *Output*: `The image shows a large, white, and very long, long, long, long, long, long, long`
- *Caption*: A wide red double-decker bus is parked beside a row of tall narrow buildings, it
  *Output*: `The image shows a bus with a red and white stripe on the side of the road, and a red and white`
- *Caption*: A tall iron gate stands at the far end of a narrow stone path, with a wooden ben
  *Output*: `The image shows the entrance to the gate, and is a very important part of the story.
The gate is`
- *Caption*: A round wooden table sits at the center of a small kitchen, with copper pots han
  *Output*: `The image shows a large, round, and very large table, with a large, round, and very large table`

**Beam (width=4):**

- *Caption*: A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall
  *Output*: `The image shows us that this is not the first time this has happened.
This is the first time this has`
- *Caption*: A bright orange and yellow maple tree stands beside a small dark pond, its round
  *Output*: `The image shows that this is the only one of its kind in the world, and that it is the only one`
- *Caption*: A wide red double-decker bus is parked beside a row of tall narrow buildings, it
  *Output*: `The image shows that the bus has been parked for over a year, and that it has been parked for over a`
- *Caption*: A tall iron gate stands at the far end of a narrow stone path, with a wooden ben
  *Output*: `The image shows that the gate was built in the middle of the night, and that it was built in the middle`
- *Caption*: A round wooden table sits at the center of a small kitchen, with copper pots han
  *Output*: `The image shows that this is the only one of its kind in the world, and that it is the only one`

### λ=+100 (→spatial)

**Greedy:**

- *Caption*: A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall
  *Output*: `The image shows the of the, the of the) and the of the) and the of the) and the`
- *Caption*: A bright orange and yellow maple tree stands beside a small dark pond, its round
  *Output*: `The image shows the of the, the of the, of the) and the of the) and the of the`
- *Caption*: A wide red double-decker bus is parked beside a row of tall narrow buildings, it
  *Output*: `The image shows the of the, the of the) and the, the of the).)))))`
- *Caption*: A tall iron gate stands at the far end of a narrow stone path, with a wooden ben
  *Output*: `The image shows the of the, the of the, of the, of the) and the of the, of`
- *Caption*: A round wooden table sits at the center of a small kitchen, with copper pots han
  *Output*: `The image shows the of the, the, of the, of the) and the, of the) and the`

**Beam (width=4):**

- *Caption*: A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall
  *Output*: `The image shows a of of of the, of the, of the, of the, of the, of the`
- *Caption*: A bright orange and yellow maple tree stands beside a small dark pond, its round
  *Output*: `The image shows the of the, the of the) and the of the) and the of the) and the`
- *Caption*: A wide red double-decker bus is parked beside a row of tall narrow buildings, it
  *Output*: `The image shows the of the, the of the) and the of the) and the of the) and the`
- *Caption*: A tall iron gate stands at the far end of a narrow stone path, with a wooden ben
  *Output*: `The image shows the of the of the, of the, of the, of the, of the, of the`
- *Caption*: A round wooden table sits at the center of a small kitchen, with copper pots han
  *Output*: `The image shows a of of of the, of the, of the, of the, of the, of the`

## `it_stands` — "It stands ___"

### λ=−100 (→descriptive)

**Greedy:**

- *Caption*: A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall
  *Output*: `It stands , however, is very very very very very very very very very very very very very very very very`
- *Caption*: A bright orange and yellow maple tree stands beside a small dark pond, its round
  *Output*: `It stands , however, is very very very very very very very very very very very very very very very very`
- *Caption*: A wide red double-decker bus is parked beside a row of tall narrow buildings, it
  *Output*: `It stands , very very very very very very very very very very very very very very very very very very very`
- *Caption*: A tall iron gate stands at the far end of a narrow stone path, with a wooden ben
  *Output*: `It stands very very very very very very very very very very very very very very very very very very very very`
- *Caption*: A round wooden table sits at the center of a small kitchen, with copper pots han
  *Output*: `It stands , and is very very very very very very very very very very very very very very very very very`

**Beam (width=4):**

- *Caption*: A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall
  *Output*: `It stands , however, is very much a very very very very very very very very very very very very very`
- *Caption*: A bright orange and yellow maple tree stands beside a small dark pond, its round
  *Output*: `It stands , however, is very much a very very very very very very very very very very very very very`
- *Caption*: A wide red double-decker bus is parked beside a row of tall narrow buildings, it
  *Output*: `It stands is very much a very very very very very very very very very very very very very very very very`
- *Caption*: A tall iron gate stands at the far end of a narrow stone path, with a wooden ben
  *Output*: `It stands , however, is very much a very very very very very very very very very very very very very`
- *Caption*: A round wooden table sits at the center of a small kitchen, with copper pots han
  *Output*: `It stands , and is very very much a very very very very very thing.

This is very very`

### λ=0 (unsteered)

**Greedy:**

- *Caption*: A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall
  *Output*: `It stands on a small hillside, and the only thing that can be said for certain is that it is`
- *Caption*: A bright orange and yellow maple tree stands beside a small dark pond, its round
  *Output*: `It stands on a small, narrow, and narrow, narrow, narrow, narrow, narrow, narrow, narrow`
- *Caption*: A wide red double-decker bus is parked beside a row of tall narrow buildings, it
  *Output*: `It stands on a narrow, narrow road, and the driver is a young man in his late thirties`
- *Caption*: A tall iron gate stands at the far end of a narrow stone path, with a wooden ben
  *Output*: `It stands at a very high point, and is covered with a thick layer of earth. The gate is a`
- *Caption*: A round wooden table sits at the center of a small kitchen, with copper pots han
  *Output*: `It stands on a small wooden table, with a small wooden stool on the far side of the room. The`

**Beam (width=4):**

- *Caption*: A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall
  *Output*: `It stands in the middle of the valley, and is covered with a thick layer of snow. The hiker`
- *Caption*: A bright orange and yellow maple tree stands beside a small dark pond, its round
  *Output*: `It stands at the end of a long, narrow path that leads to a small pond. The pond has a`
- *Caption*: A wide red double-decker bus is parked beside a row of tall narrow buildings, it
  *Output*: `It stands at the end of a long, narrow corridor, with a long, narrow staircase leading down to a`
- *Caption*: A tall iron gate stands at the far end of a narrow stone path, with a wooden ben
  *Output*: `It stands at the end of a narrow stone path, with a wooden bench placed directly in front of it and`
- *Caption*: A round wooden table sits at the center of a small kitchen, with copper pots han
  *Output*: `It stands at the end of a long, narrow corridor that leads to the kitchen.
This is where the`

### λ=+100 (→spatial)

**Greedy:**

- *Caption*: A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall
  *Output*: `It stands at of the, of the, of the) and the of the) and the of the)`
- *Caption*: A bright orange and yellow maple tree stands beside a small dark pond, its round
  *Output*: `It stands at of, of, of, of, of, of, of, of, of, of`
- *Caption*: A wide red double-decker bus is parked beside a row of tall narrow buildings, it
  *Output*: `It stands on of the, the of the) and the of the) and the of the) and the`
- *Caption*: A tall iron gate stands at the far end of a narrow stone path, with a wooden ben
  *Output*: `It stands on of the, of the, of the, of the, of the, of the) and`
- *Caption*: A round wooden table sits at the center of a small kitchen, with copper pots han
  *Output*: `It stands on of, of, of, of, of, of, of, of, of, of`

**Beam (width=4):**

- *Caption*: A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall
  *Output*: `It stands , of the, of the, of the, of the, of the, of the, of`
- *Caption*: A bright orange and yellow maple tree stands beside a small dark pond, its round
  *Output*: `It stands , of the, of the, of the, of the, of the, of the, of`
- *Caption*: A wide red double-decker bus is parked beside a row of tall narrow buildings, it
  *Output*: `It stands on of, of, of, of, of, of, of, of, of, of`
- *Caption*: A tall iron gate stands at the far end of a narrow stone path, with a wooden ben
  *Output*: `It stands on of, of, of, of, of, of, of, of, of, of`
- *Caption*: A round wooden table sits at the center of a small kitchen, with copper pots han
  *Output*: `It stands , of the, of the, of the, of the, of the, of the, of`

## `it_appears` — "It appears ___"

### λ=−100 (→descriptive)

**Greedy:**

- *Caption*: A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall
  *Output*: `It appears very very very very very very very very very very very very very very very very very very very very`
- *Caption*: A bright orange and yellow maple tree stands beside a small dark pond, its round
  *Output*: `It appears very very very very very very very very very very very very very very very very very very very very`
- *Caption*: A wide red double-decker bus is parked beside a row of tall narrow buildings, it
  *Output*: `It appears very very very very very very very very very very very very very very very very very very very very`
- *Caption*: A tall iron gate stands at the far end of a narrow stone path, with a wooden ben
  *Output*: `It appears very very very very very very very very very very very very very very very very very very very very`
- *Caption*: A round wooden table sits at the center of a small kitchen, with copper pots han
  *Output*: `It appears very very very very very very very very very very very very very very very very very very very very`

**Beam (width=4):**

- *Caption*: A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall
  *Output*: `It appears , however, is a very very very very very very very very very very very very very very very`
- *Caption*: A bright orange and yellow maple tree stands beside a small dark pond, its round
  *Output*: `It appears , however, is a very very very very very very very very very very very very very very very`
- *Caption*: A wide red double-decker bus is parked beside a row of tall narrow buildings, it
  *Output*: `It appears , however, is a very very very very very very very very very very very very very very very`
- *Caption*: A tall iron gate stands at the far end of a narrow stone path, with a wooden ben
  *Output*: `It appears , however, is a very very very very very very very very very very very very very very very`
- *Caption*: A round wooden table sits at the center of a small kitchen, with copper pots han
  *Output*: `It appears , however, is a very very very very very very very very very very very very very very very`

### λ=0 (unsteered)

**Greedy:**

- *Caption*: A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall
  *Output*: `It appears to be a small town, but it is actually a small town, and it is a very small`
- *Caption*: A bright orange and yellow maple tree stands beside a small dark pond, its round
  *Output*: `It appears to be a large, round, and very long, long, long, long, long, long`
- *Caption*: A wide red double-decker bus is parked beside a row of tall narrow buildings, it
  *Output*: `It appears to be a small, but very well-kept, house. The bus is parked in a corner`
- *Caption*: A tall iron gate stands at the far end of a narrow stone path, with a wooden ben
  *Output*: `It appears to be a large, well-built building, with a large number of rooms and a large number`
- *Caption*: A round wooden table sits at the center of a small kitchen, with copper pots han
  *Output*: `It appears to be a large, round, and very old, and has a very long, narrow, and`

**Beam (width=4):**

- *Caption*: A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall
  *Output*: `It appears to be the last of its kind on this side of the valley, and the only one of its`
- *Caption*: A bright orange and yellow maple tree stands beside a small dark pond, its round
  *Output*: `It appears to be a small pond, but it is actually a large one. The pond is covered with a`
- *Caption*: A wide red double-decker bus is parked beside a row of tall narrow buildings, it
  *Output*: `It appears to have been built in the late 19th and early 20th centuries, and has been used by`
- *Caption*: A tall iron gate stands at the far end of a narrow stone path, with a wooden ben
  *Output*: `It appears to have been built in the middle of the last century or so, and has been used by the`
- *Caption*: A round wooden table sits at the center of a small kitchen, with copper pots han
  *Output*: `It appears to have been built in the late 19th century, and it is said to have been used as`

### λ=+100 (→spatial)

**Greedy:**

- *Caption*: A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall
  *Output*: `It appears to be the of the, of the, of the, of the) and the of the)`
- *Caption*: A bright orange and yellow maple tree stands beside a small dark pond, its round
  *Output*: `It appears to be the of the, the of the, the, of the) and the of the)`
- *Caption*: A wide red double-decker bus is parked beside a row of tall narrow buildings, it
  *Output*: `It appears to of the, the of the, the of the) and the of the) and the of`
- *Caption*: A tall iron gate stands at the far end of a narrow stone path, with a wooden ben
  *Output*: `It appears to of the, of the, of the, of the, of the, of the) and`
- *Caption*: A round wooden table sits at the center of a small kitchen, with copper pots han
  *Output*: `It appears to be the of the, the of the, the, the, the, the, the,`

**Beam (width=4):**

- *Caption*: A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall
  *Output*: `It appears to be the of the, of the, of the, of the, of the, of the`
- *Caption*: A bright orange and yellow maple tree stands beside a small dark pond, its round
  *Output*: `It appears to be the of the, of the, of the, of the, of the, of the`
- *Caption*: A wide red double-decker bus is parked beside a row of tall narrow buildings, it
  *Output*: `It appears to be the of the, of the, of the, of the, of the, of the`
- *Caption*: A tall iron gate stands at the far end of a narrow stone path, with a wooden ben
  *Output*: `It appears to be the of the, of the, of the, of the, of the, of the`
- *Caption*: A round wooden table sits at the center of a small kitchen, with copper pots han
  *Output*: `It appears to be the of the, of the, of the, of the, of the, of the`
