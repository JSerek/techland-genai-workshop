# Golden dataset — propozycja do weryfikacji (20 negatywnych recenzji, multi-label)

**Źródło:** `data/raw/dying_light_beast_negative_full.csv` (english, negative).
**Dobór:** pas długości 200–700 znaków, ranking po `votes_up`, z wymuszonym pokryciem 15 aspektów
oraz przewagą komentarzy wielo-etykietowych. **Taksonomia:** `analysis/aspects_with_definitions.md`.

> **ZASADA (D-007): klasyfikujemy wyłącznie NEGATYWNE aspekty.** Jeśli recenzent coś chwali
> (np. „storytelling is the best", „parkour improved"), tej etykiety **nie** nadajemy — liczy się
> tylko to, co krytykuje. **Brak `franchise_frame` (D-008):** werdykt „to powinno być DLC / to nie
> pełna gra" trafia do `price`. **Nowa kategoria `gore`** (krew, miażdżone kości, efekty obrażeń).
>
> **Zmiany względem poprzedniej wersji:** podmieniono 2 komentarze, by utrzymać pokrycie po
> wprowadzeniu reguły negative-only (audio i night_horror były wcześniej ujęte pozytywnie):
> usunięto `209560256` i `204629841`; dodano `207836299` (night_horror) i `215810589` (audio).

**Pokrycie aspektów (liczba komentarzy):**
combat 10 · parkour 4 · enemies 5 · night_horror 1 · progression 6 · world 6 · story 8 ·
bugs 3 · performance 2 · graphics 2 · audio 1 · content 8 · price 9 · coop 1 · gore 2

*(night_horror, audio, coop mają po 1 — w recenzjach negatywnych występują rzadko.)*

---

### 1 · review_id 205490944 · votes_up 144
> I very much enjoyed the 30+ hours... but this is not a game. This is a DLC pack for Dying Light 2 that they shoved a bunch of filler into. It's fun... but I can't recommend it at it's current price point.

**LABELS:** `content`, `price`
**Dlaczego:** „filler" → content; „not a game / DLC pack … at its current price point" (werdykt DLC = wartość) → price. „Enjoyed / fun" = pochwały, pomijamy.

### 2 · review_id 204662234 · votes_up 131
> Game just doesn't feel fun, Zombies are to tanky, weapons are made of toilet paper, animation locks from zombies... mechanics were shallow... feels like a DLC... you can't pass this off as a full game.

**LABELS:** `combat`, `enemies`, `price`
**Dlaczego:** „weapons toilet paper / animation locks / shallow mechanics" → combat; „zombies too tanky" (balans wrogów) → enemies; „feels like a DLC / can't pass off as a full game" → price.

### 3 · review_id 204658386 · votes_up 119
> ...meant to be a DLC... the price definitely isn't worth it... skill tree and the progression very early on that this game would be extremely short... short experience for a game that is worth 60 dollars. I would definitely buy this if it was 40 dollars or less.

**LABELS:** `progression`, `content`, `price`
**Dlaczego:** „skill tree and progression" → progression; „extremely short" → content; „60 / 40 dollars / DLC / not worth" → price. „Upgrade from original and DL2" = pochwała, pomijamy.

### 4 · review_id 204958301 · votes_up 99
> The Grab mechanic the zombies have ruins this game. they grab way too often, and it even cancels you if you are mid swing with your melee attack... it just grabbed me any way... suuuper annoying.

**LABELS:** `combat`, `enemies`
**Dlaczego:** przerywanie ataku melee / feel walki → combat; częstotliwość i mechanika grab wrogów → enemies. *(uwaga prowadzącego #2)*

### 5 · review_id 217659097 · votes_up 86
> They make you travel five minutes to talk to someone for two seconds, then travel five minutes back... kill a few zombies, and rinse and repeat that silly loop for 80% of the game.

**LABELS:** `world`, `content`
**Dlaczego:** bieganie A→B, struktura questów/brak fast travel → world; „rinse and repeat 80%" → content.

### 6 · review_id 204834661 · votes_up 70
> ...it's lost the charm the first game had. its very repetitive, combat start to get monotonous, and zombies non stop grapple. i'd give it a 5.5/10

**LABELS:** `content`, `combat`, `enemies`
**Dlaczego:** „very repetitive" → content; „combat monotonous" → combat; „zombies non stop grapple" → enemies. *(uwaga prowadzącego #3)*

### 7 · review_id 204957991 · votes_up 51
> For 60 dollars, you're getting: a DLC rushed into a full game... dying light 2 with less features, less exploration, and less variety. Yay Crane is back, filled with plot holes!... cars are back, but they're slow... guns... even on brutal, why use anything else!

**LABELS:** `price`, `content`, `world`, `story`, `combat`
**Dlaczego:** „60 dollars / DLC at full price" → price; „less features/variety" → content; „less exploration / cars slow / area levels" → world; „Crane, plot holes" → story; „guns trivialize even on brutal" (balans broni) → combat.

### 8 · review_id 204639308 · votes_up 31
> Reused a ton of assets then charging $60... Combat is somehow less responsive than DL2, and parkour feels float-y and slow. Skill tree is extremely simplified... barren... especially for a dlc priced at a triple A game price.

**LABELS:** `combat`, `parkour`, `progression`, `price`
**Dlaczego:** „combat less responsive" → combat; „parkour floaty/slow" → parkour; „skill tree simplified/barren" → progression; „$60 / dlc priced AAA" → price.

### 9 · review_id 204632770 · votes_up 28
> ...cons -Art Direction, colors... look worse... -Bullet sponge everywhere -No scaling... -Dead Island 2 has a better gore system, and way better feeling weapons -Still no impact on melee... -Asset flipping DL2, animations... UI design -Selling a DL2 DLC... for 60. -No parkour points -All kinds of bugs for example in parkour it's quite obvious it's not polished.

**LABELS:** `graphics`, `combat`, `gore`, `progression`, `bugs`, `price`
**Dlaczego:** „Art Direction/colors worse, animations, UI" → graphics; „bullet sponge / no impact on melee" → combat; „Dead Island 2 better gore system" → gore; „No parkour points / no scaling" (ekonomia XP) → progression; „all kinds of bugs / not polished" → bugs; „DL2 DLC for 60" → price. „Runs smooth / parkour less floaty / beast fun / OST DL1 vibes" = pochwały, pomijamy.

### 10 · review_id 204870501 · votes_up 27
> It's a really well done, high quality game, the storytelling is the best out of all Dying Light entries, but... there's just not enough content to justify the price tag... The Beast doesn't give a reason to continue playing after the main story... get this game when it's discounted.

**LABELS:** `content`, `price`
**Dlaczego:** „not enough content / nothing to do after main story" → content; „price tag / discounted" → price. **story pominięte — było wyraźną pochwałą** *(uwaga prowadzącego #4)*.

### 11 · review_id 206740572 · votes_up 23
> i love the dying light series, but this one feels kind of a letdown. there's definitely an improvement from dying light 2, parkour and combat, though the same can't be said about the story. the game is so SHORT... it feels unfinished... I'd recommend getting it on sale.

**LABELS:** `story`, `content`, `price`
**Dlaczego:** „story" (krytyka) → story; „so SHORT / unfinished" → content; „on sale" → price. **parkour i combat pominięte — wspomniane pozytywnie** *(uwaga prowadzącego #5)*.

### 12 · review_id 204878716 · votes_up 21
> Dying Light: The Buggy Beast... - Missions that cannot be completed/started - Game crashes - Fps drops... - Night/Day cycle completely bugged - Falling through the map - Teleportation glitches - Rain indoor...

**LABELS:** `bugs`, `performance`
**Dlaczego:** crashe/glitche/softlocki → bugs; „Fps drops" → performance.

### 13 · review_id 204642036 · votes_up 19
> ...it's written worse than dying light 2, the gameplay is messier, the optimization sucks i constantly dropped below 20 fps on low with a 3070 and a ryzen 9 5900x...

**LABELS:** `story`, `performance`
**Dlaczego:** „written worse" → story; „optimization sucks / <20 fps" → performance.

### 14 · review_id 205076487 · votes_up 18
> Game is boring and story isn't well written, 80% of the game is go from point A to B... It's a crime this game doesn't have fast travel. Combat is bad and sloppy, and movement could use a lot of work...

**LABELS:** `story`, `world`, `combat`, `parkour`
**Dlaczego:** „story isn't well written" → story; „A to B / no fast travel" → world; „combat bad and sloppy" → combat; „movement could use work" → parkour. „Ending good" = pochwała, pomijamy.

### 15 · review_id 205100534 · votes_up 13
> The world is literally dead... there is no life in it... you don't encounter them except during missions... For a AAA game from 2025, such a cardboard-like world is very noticeable.

**LABELS:** `world`
**Dlaczego:** projekt/wiarygodność świata, brak życia, pusta eksploracja → world. (Czysty single-label.)

### 16 · review_id 204641037 · votes_up 11
> Inferior in every way to the old dying light. Even it's graphics are worse, you can count the pixels in blood splatters, the gore is just ripped off from the second game too. Skills are bland and boring. Also they turned Kyle Crane into a ''run when my eyes turn red'' anime character.

**LABELS:** `graphics`, `gore`, `progression`, `story`
**Dlaczego:** „graphics worse" → graphics; „pixels in blood splatters / gore ripped off" → gore; „skills bland and boring" → progression; „Kyle Crane … anime character" → story.

### 17 · review_id 204926978 · votes_up 8
> Skill tree was mid... Parkour + combat felt jank... Grapple hook was terrible... Side quests didn't feel repetitive thankfully, but there definitely wasn't enough of them... Kyle's dialogue still feels like he's a 10 year old child... I'd honestly recommend this game if there was just more content.

**LABELS:** `progression`, `combat`, `parkour`, `story`, `content`
**Dlaczego:** „skill tree mid" → progression; „combat jank" → combat; „parkour jank / grapple hook terrible" → parkour; „Kyle's dialogue childish" → story; „not enough / more content" → content. **Music/gun play/throwables/vehicles/nighttime chases = pochwały, pomijamy** *(reguła #4)*.

### 18 · review_id 205334075 · votes_up 3
> ...Within 10 minutes of having unlocked CO-op, me and my friend got stuck on a bridge... we were somehow able to hit enemies through the wall getting immense amounts of XP... beat the boss without him even responding. This game will need 'ALOT' of fixes.

**LABELS:** `coop`, `bugs`
**Dlaczego:** sesja co-op (stuck na moście, gra we dwóch) → coop; hity przez ścianę / boss bez reakcji / „needs a lot of fixes" → bugs.

### 19 · review_id 207836299 · votes_up 8 · *(NOWY — pokrycie night_horror)*
> It's not fun... The map design seems to forget that this series is a silly parkour... most of it is just open terrain with little verticality. When night rolls around you can either hide or get killed by the numerous infected that seem to spawn on you just as you get out of sight. The combat is meaty... and then constant zombie grapples and bosses with needlessly large health bars sapped any love I had for fighting... the story is bland and un-inspired. Wait for a good sale or just play the first one.

**LABELS:** `night_horror`, `world`, `enemies`, `combat`, `story`, `price`
**Dlaczego:** „when night rolls around you can either hide or get killed / infected spawn on you" → night_horror; „open terrain, little verticality, map design" → world; „infected spawn on you / zombie grapples" → enemies; „bosses needlessly large health / sapped love for fighting" → combat; „story bland un-inspired" → story; „wait for a good sale" → price. „Combat meaty nice feel for a time" = częściowa pochwała, ale ostatecznie skrytykowana.

### 20 · review_id 215810589 · votes_up 11 · *(NOWY — pokrycie audio)*
> ...They didn't even try to create a story, the Antagonist is a Far Cry-esque cartoon cutout, parcour is clunky, combat feels outdated, "bosses" are damage-sponges with 3 attack patterns each, crafting relies on you collecting thousands of garbage items, dark zones are worthless copy paste dungeons... But hey, Kyle Crane is back, with a voice actor who's half asleep while reading his lines.

**LABELS:** `story`, `audio`, `parkour`, `combat`, `enemies`, `progression`, `world`
**Dlaczego:** „didn't even try to create a story / antagonist cartoon" → story; „voice actor half asleep reading lines" → audio; „parcour clunky" → parkour; „combat outdated" → combat; „bosses damage-sponges 3 patterns" → enemies; „crafting collecting thousands of garbage items" → progression; „dark zones worthless copy paste dungeons" → world.
