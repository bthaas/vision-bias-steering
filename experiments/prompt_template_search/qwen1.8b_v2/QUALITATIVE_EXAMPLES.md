# Qualitative Generation Examples — Qwen-1.8B

Shown at λ=−100, λ=0, λ=+100. Greedy first, then beam (width=4).

## `A2_in_the` (Approach A2) — prefix: "[caption]. In the"

### λ=−100 (→descriptive)
**Greedy:**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `[caption]. In the no one a one a no a no a no a no a no a no a no a no`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `[caption]. In the no one a only a one a no a no a no a no a no a no a no`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `[caption]. In the no one a only one a no one a no one a no no a no no a no no`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `[caption]. In the no one a only a one a no a no a no a no a no a no a no`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `[caption]. In the no one a有一个只有一个有一个还有一个似乎有一个有一个有一个有一个有一个有一个有一个有一个有一个有一个有一个有一个`

**Beam (width=4):**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `[caption]. In the a no a no a no a no a no a no a no a no a no a no`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `[caption]. In the no a a no a no a no a no a no a no a no a no a no`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `[caption]. In the a no a no a no a no a no a no a no a no a no a no`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `[caption]. In the a no a no a no a no a no a no a no a no a no a no`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `[caption]. In the a no a no a no a no a no a no a no a no a no a no`

### λ=0 (no steering)
**Greedy:**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `[caption]. In the distance, a mountain range can be seen towering over the horizon. The hiker's face is塑造`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `[caption]. In the distance, a tall white building can be seen with a large window that looks out onto the pond.`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `[caption]. In the distance, a cityscape can be seen through the trees, with towering skyscrapers and busy streets`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `[caption]. In the distance, a small village can be seen nestled against the horizon, with buildings made of thatched roofs`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `[caption]. In the corner of the room, a vintage stove is visible, its hearth covered in ashes. The walls`

**Beam (width=4):**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `[caption]. In the distance, a mountain range can be seen in the distance, with snow-capped peaks towering above the`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `[caption]. In the distance, the trees of a nearby forest can be seen, their leaves rustling in the gentle breeze`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `[caption]. In the distance, the city skyline can be seen, with towering skyscrapers reaching towards the sky. The`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `[caption]. In the center of the path is a small pond, surrounded by lush greenery and dotted with wildflowers.`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `[caption]. In the center of the table, a small wooden cutting board is propped up, with a knife and a`

### λ=+100 (→spatial)
**Greedy:**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `[caption]. In the ,,,,,,,,,,,,,,,,,,,,`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `[caption]. In the ,,,,,,,,,,,,,,,,,,,,`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `[caption]. In the ,,,,,,,,,,,,,,,,,,,,`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `[caption]. In the ,,,,,,,,,,,,,,,,,,,,`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `[caption]. In the ,,,,,,,,,,,,,,,,,,,,`

**Beam (width=4):**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `[caption]. In the ,,,,,,,,,,,,,,,,,,,,`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `[caption]. In the ,,,,,,,,,,,,,,,,,,,,`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `[caption]. In the ,,,,,,,,,,,,,,,,,,,,`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `[caption]. In the ,,,,,,,,,,,,,,,,,,,,`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `[caption]. In the ,,,,,,,,,,,,,,,,,,,,`

## `A_in_the` (Approach A) — prefix: "In the"

### λ=−100 (→descriptive)
**Greedy:**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `In the no one a only a one a no one a no a no a no a no a no a`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `In the no one a one a no one a no one a no one a no one a no one a`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `In the no one a one a no one a no one a no one a no one a no one a`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `In the no one a only a one a no a no a no a no a no a no a no`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `In the no one a one a no one a no one a no one a no one a no one a`

**Beam (width=4):**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `In the a one a no a no a no a no a no a no a no a no a no`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `In the a one a no a no a no a no a no a no a no a no a no`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `In the a one a no a no a no a no a no a no a no a no a no`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `In the a no a no a no a no a no a no a no a no a no a no`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `In the a no a no a no a no a no a no a no a no a no a no`

### λ=0 (no steering)
**Greedy:**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `In the distance, there is a mountain range in the background, with peaks towering above the hiker. The`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `In the distance, there is a small village with houses made of wood and stone, surrounded by lush green fields`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `In the image, a large and impressive double-decker bus stands next to a row of towering, narrow buildings`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `In the distance, there is a tall iron gate standing at the end of a narrow stone path. The gate`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `In the image described, there is a round wooden table located in the center of a small kitchen. The table`

**Beam (width=4):**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `In the image, a lone hiker can be seen standing on top of a snow-dusted ridge, with`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `In the image of a bright orange and yellow maple tree standing beside a small dark pond, the tree appears to`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `In the image of a wide red double-decker bus parked beside a row of tall narrow buildings, the bus`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `In the image, there is a tall iron gate standing at the far end of a narrow stone path. The`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `In the image, there is a small kitchen with a round wooden table in the center. The table is surrounded`

### λ=+100 (→spatial)
**Greedy:**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `In the ,,,,,,,,,,,,,,,,,,,,`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `In the ,,,,,,,,,,,,,,,,,,,,`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `In the ,,,,,,,,,,,,,,,,,,,,`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `In the ,,,,,,,,,,,,,,,,,,,,`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `In the ,,,,,,,,,,,,,,,,,,,,`

**Beam (width=4):**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `In the ,,,,,,,,,,,,,,,,,,,,`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `In the ,,,,,,,,,,,,,,,,,,,,`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `In the ,,,,,,,,,,,,,,,,,,,,`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `In the ,,,,,,,,,,,,,,,,,,,,`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `In the ,,,,,,,,,,,,,,,,,,,,`

## `B_positioned` (Approach B) — prefix: "Positioned"

### λ=−100 (→descriptive)
**Greedy:**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `Positioned a one a no a a no a no a no a no a no a no a no a`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `Positioned a one a no a one a no a no a no a no a no a no a no`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `Positioned a one a no a no a no a no a no a no a no a no a no`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `Positioned a one a no a a no a a no a no a no a no a no a no`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `Positioned a one a no a a no a a no a a no a a no a a no a`

**Beam (width=4):**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `Positioned a one a one a no a no a no a no a no a no a no a no`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `Positioned a one a one a no a no a no a no a no a no a no a no`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `Positioned a a no a no a no a no a no a no a no a no a no a`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `Positioned a a no a a no a no a no a no a no a no a no a no`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `Positioned a a no a a no a no a no a no a no a no a no a no`

### λ=0 (no steering)
**Greedy:**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `Positioned at the edge of a vast, snowy landscape, a solitary figure can be seen standing atop a rocky`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `Positioned in a serene natural setting, this image captures the beauty of autumn's colors. The maple tree is`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `Positioned in a busy urban area, this image captures the essence of a bustling city street. The bus itself`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `Positioned on the far side of a narrow stone path, a tall iron gate stands as the focal point of`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `Positioned in a cozy and intimate setting, this image captures a sense of warmth and comfort. The wooden table`

**Beam (width=4):**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `Positioned on a snow-covered ridge, the lone hiker stands with their back to the viewer, gazing`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `Positioned in a serene and natural setting, this image captures the beauty and tranquility of a deciduous maple`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `Positioned in a busy urban environment, this image captures the essence of a bustling street scene. The bus,`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `Positioned at the far end of a narrow stone path, this image depicts a scene of tranquility and ser`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `Positioned at the center of a cozy kitchen, the round wooden table is the focal point of the scene.`

### λ=+100 (→spatial)
**Greedy:**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `Positioned ,,,,,,,,,,,,,,,,,,,,`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `Positioned ,,,,,,,,,,,,,,,,,,,,`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `Positioned ,,,,,,,,,,,,,,,,,,,,`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `Positioned ,,,,,,,,,,,,,,,,,,,,`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `Positioned ,,,,,,,,,,,,,,,,,,,,`

**Beam (width=4):**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `Positioned ,,,,,,,,,,,,,,,,,,,,`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `Positioned ,,,,,,,,,,,,,,,,,,,,`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `Positioned ,,,,,,,,,,,,,,,,,,,,`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `Positioned ,,,,,,,,,,,,,,,,,,,,`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `Positioned ,, of,,,,,,,,,,,,,,,,,`

## `A2_main_subject` (Approach A2) — prefix: "[caption]. The main subject appears"

### λ=−100 (→descriptive)
**Greedy:**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `[caption]. The main subject appears no one a only a one a no a no a no a no a no a no a no`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `[caption]. The main subject appears no one a only a one a one a no a no a no a no a no a no`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `[caption]. The main subject appears no one a only one a no a no no a no a no a no a no a no`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `[caption]. The main subject appears no one a only a one a no a no a no a no a no a no a no`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `[caption]. The main subject appears no one a only a one a no a no a no a no a no a no a no`

**Beam (width=4):**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `[caption]. The main subject appears a one a one a one a no a no a no a no a no a no a no`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `[caption]. The main subject appears a one a one a one a no a no a no a no a no a no a no`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `[caption]. The main subject appears only one a no a no a no a no a no a no a no a no a no`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `[caption]. The main subject appears a one a one a no a no a no a no a no a no a no a no`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `[caption]. The main subject appears a one a one a no a no a no a no a no a no a no a no`

### λ=0 (no steering)
**Greedy:**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `[caption]. The main subject appears to be the hiker, who is wearing warm clothing and carrying a backpack filled with supplies. Despite`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `[caption]. The main subject appears to be the tree, with its branches reaching out towards the sky. In the background, there is`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `[caption]. The main subject appears to be the bus itself, with its large windows letting in plenty of natural light. The driver's`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `[caption]. The main subject appears to be a woman, dressed in a flowing white dress that billows slightly as she walks. Her`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `[caption]. The main subject appears to be a woman in her mid-40s, wearing a white apron over a dark`

**Beam (width=4):**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `[caption]. The main subject appears to be the hiker, with their back turned to the viewer and their head held high. The`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `[caption]. The main subject appears to be the tree, with its trunk and branches reaching towards the sky. In the background, there`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `[caption]. The main subject appears to be the bus itself, with its large windows letting in the warm glow of the setting sun.`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `[caption]. The main subject appears to be a woman, dressed in a flowing white dress that billows slightly in the breeze. She`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `[caption]. The main subject appears to be a woman in her mid-30s, wearing a white apron and carrying a`

### λ=+100 (→spatial)
**Greedy:**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `[caption]. The main subject appears ,,,,,,,,,,,,,,,,,,,,`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `[caption]. The main subject appears ,,,,,,,,,,,,,,,,,,,,`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `[caption]. The main subject appears ,,,,,,,,,,,,,,,,,,,,`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `[caption]. The main subject appears ,,,,,,,,,,,,,,,,,,,,`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `[caption]. The main subject appears ,,,,,,,,,,,,,,,,,,,,`

**Beam (width=4):**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `[caption]. The main subject appears ,,,,,,,,,,,,,,,,,,,,`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `[caption]. The main subject appears ,,,,,,,,,,,,,,,,,,,,`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `[caption]. The main subject appears ,,,,,,,,,,,,,,,,,,,,`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `[caption]. The main subject appears ,,,,,,,,,,,,,,,,,,,,`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `[caption]. The main subject appears ,,,,,,,,,,,,,,,,,,,,`

## `A_subject_looks` (Approach A) — prefix: "The subject looks"

### λ=−100 (→descriptive)
**Greedy:**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `The subject looks no one a似乎有一个只有一个一个有一个有一个还有一个有一个有一个有一个有一个有一个有一个有一个有一个有一个有一个`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `The subject looks no one a only a a no a no a no a no a no a no a no a`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `The subject looks no one a似乎有一个只有一个项 a no a no no a no a no no no a no`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `The subject looks no one a似乎有一个只有一个一个没有一个是总有一个有一个还有一个有一个有一个有一个有一个有一个有一个有一个`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `The subject looks no one a似乎有一个只有一个一个有一个有一个还有一个有一个有一个有一个有一个有一个有一个有一个有一个有一个有一个`

**Beam (width=4):**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `The subject looks only one a one a one a one a no one a one a one a no one a no`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `The subject looks only one a one a one a one a no one a one a one a no one a no`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `The subject looks no one a a no a no a no a no a no a no a no a no a`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `The subject looks a one a no a a no a no a no a no a no a no a no a`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `The subject looks no a one a no a no a no a no a no a no a no a no a`

### λ=0 (no steering)
**Greedy:**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `The subject looks out at a vast expanse of snow-covered terrain stretching out as far as the eye can see.`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `The subject looks like it is in a peaceful and serene environment, with a bright and cheerful atmosphere. The maple tree`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `The subject looks like a busy city street scene with a busy bus and a row of tall buildings. The bus appears`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `The subject looks out over a small, enclosed space, perhaps a garden or yard. The gate is made of iron`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `The subject looks at the scene from a low angle, as if trying to capture all the details in one shot.`

**Beam (width=4):**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `The subject looks out over a vast expanse of snow-covered terrain, with the distant mountains in the background. The`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `The subject looks serene and peaceful, surrounded by nature's beauty. The bright orange and yellow maple tree stands tall and`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `The subject looks like a busy street scene, with a busy bus and a row of tall narrow buildings. The bus`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `The subject looks out over a peaceful landscape, with a tall iron gate standing at the far end of a narrow stone`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `The subject looks cozy and inviting, with a warm and welcoming atmosphere. The wooden table is the focal point of the`

### λ=+100 (→spatial)
**Greedy:**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `The subject looks ,,,,,,,,,,,,,,,,,,,,`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `The subject looks ,,,,,,,,,,,,,,,,,,,,`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `The subject looks ,,,,,,,,,,,,,,,,,,,,`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `The subject looks ,,,,,,,,,,,,,,,,,,,,`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `The subject looks ,,,,,,,,,,,,,,,,,,,,`

**Beam (width=4):**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `The subject looks ,,,,,,,,,,,,,,,,,,,,`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `The subject looks ,,,,,,,,,,,,,,,,,,,,`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `The subject looks ,,,,,,,,,,,,,,,,,,,,`
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `The subject looks ,,,,,,,,,,,,,,,,,,,,`
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `The subject looks ,,,,,,,,,,,,,,,,,,,,`
