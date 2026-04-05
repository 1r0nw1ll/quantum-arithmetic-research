---
source: QA-Vol-II-books-3-&-4/00-QA_Books_3_&_4-ALL-PAGES.docx
series: QA-4
method: docx
chars: 228720
extracted: 2026-03-26
---

# QA-4 — 00-QA_Books_3_&_4-ALL-PAGES

Quantum Arithmetic - Volume II
Books 3 & 4
New Wave Theory - Synchronous Harmonics
by Ben Iverson
February, 1991
(long lost book - now found 11/10/05)
© Delta Spectrum Research, 2005
Published by
Delta Spectrum Research
921 Santa Fé Avenue
La Junta, Colorado 81050
Web Site:
http://www.svpvril.com
Ben Iverson also authored:
Foundation of Science
Global Geometry
Great Flood Mystery
Pythagoras and the Quantum World - Vol. I
Pythagoras and the Quantum World - Vol 2
Pythagoras and the Quantum World - Vol 2 revised
Pythagoras and the Quantum World - Vol 3 Enneagram
Pythagoras and the Quantum World - Vol. II
QA-1 Natural Arithmetic
QA-2 Natural Arithmetic
QA-3 Natural Arithmetic
Quantum Arithmetic - 1 & 2
Quantum Arithmetic - Book 3 & 4 - New Wave Theory - Synchronous Harmonics Quantum Arithmetic 2nd Keely Conference
Quantum Arithmetic 2nd Keely Conference
Quantum Arithmetic 3rd Keely Conference
Quantum Arithmetic 3rd Keely Conference
Quantum Arithmetic Reference
Synchronous Harmonics
and numerous articles. Some are located here:
http://www.svpvril.com/svpweb17.html
Quantize Code
optimized by Dale Pond for HyperCard but will run in Revolution with minimal modification.
on mouseUp
set cursor to watch
put "" into background field b
put "" into background field e
put "" into background field d
put "" into background field a
put "" into background field bb
put "" into background field ee
put "" into background field dd
put "" into background field aa
put "" into card field id 42
put "" into card field id 43
put "" into field id 20
if card field Jo 1 th en
ask "Please enter decimal only"
put it into card field Jo
end if
put (10000) into xxk
put card field Jo into x1
put the value of x1*10000 into xxj  put (xxk-xxj)/2 into xxc
put (xxk + xxj)/2 into xxd
put xxd^.5 into xd
put xxj/xd into xb
put xxc/xd into xe
put xxk/xd into xa
put 1/xd into ratio
put ratio*xb into b
put ratio*xe into e
put ratio*xd into d
put ratio*xa into a
put (0) into xx1
put 0.00005 into x2 --allowed empirical error
--FIND QUANTUM RATIO: (Euclid 7,28)  --EXPAND EMPIRICAL SQUARE TO QUAN TUM SQUARE
repeat while xy <> yx
put xx1 + 1 into xx1
put xx1 into field id 11
put ratio into x
put xx1 * b into bb
put xx1 * e into ee
put xx1 * d into dd
put xx1 * a into aa
put abs(round(aa + 0.5 - x2)) into xy  put abs(round(aa - 0.5 + x2)) into yx  if xy = yx then
exit repeat
end if
next repeat
end repeat
--CHECK PRIMENESS (Euclid 7,1)
put abs(round(bb)) into bbb
put abs(trunc(xy)) into ddd
Repeat until bbb=ddd
put abs(ddd) into field id 9
put abs(bbb) into field id 8
If ddd>bbb then
put ddd-bbb into ddd
end if
If bbb>ddd then
put bbb-ddd into bbb
put bbb into field id 12
put ddd into field id 13
next repeat
end if
put bbb into field id 12
put ddd into field id 13
if bbb=1 and ddd =1 then
put bbb into field id 14
put ddd into field id 15
exit repeat
end if
end repeat
--SET SUM & DIFFERENCE: (Fibonacci- Dio phantus)
put abs(round((dd)/(ddd))) into field id 10  put abs(trunc((xy))/(bbb)) into field id 11  put abs(trunc(field id 11 - field id 10)) into field id 9
put abs(trunc(field id 10 - field id 9)) into field id 8
--EXTEND SERIES: (Rhind Mathematical Pa pyrus)
put trunc(field id 11) into card field id 42  put card field Jo into x1
put the value of x1 * card field id 42 into card field id 43
put round(card field id 43) into card field id 43
put card field id 43/card field id 42 into field id 20
put abs(field id 11) into field id 12  put abs(field id 9 * field id 10) into field id 14  put abs(field id 14 - field id 12) into field id 13
put field id 13 + field id 14 into field id 15  end mouseup
INTRODUCTION
Additional background for Books #3 and #4 is given in Volumes I, II, & III, of "Pythagoras And The Quantum World", (1982-1986).
Volume I should be a prerequisite for this volume, but the reader will de rive some idea of general goals to be reached.
CONTENTS
BOOK 3 - Part I
1 Illusions
2 To Be Prime
4 Test
5 Time Synchronization Energy in Matter
6 Even Numbers
Graphics
7 Euclid’s Four Number Types Wave Packets
8 Phasing In
9 Problems & Questions Multiple Waves
10 The Harmonic Cycle
Other Cycles
11 Phasing In (again)
Exception
12 Most Basic Multiple Cycle Complex Cycles
Synchronous Points
Harmonic Points
Wave Packets
13 Aliquot Parts
Example
Questions
Harmonics
14 The Mathematics
15 Interdependence
Wave Packets
Phasing (again)
Examples
Physical Harmonics
16 Theory
Various Myriads of Harmony 17 Theory of Harmony
Theory of Energy Forms
Experimentation
18 Harmony
Cattle Problem
I-Ching
19 Not Archimedes?
Quantum Pitch
Cascades
The Senses
BOOK 3 - Part II
20 Music of the Spheres A Musical Scale?
History
And I-Ching
21 Beats
Compositions
22 Pitch
Poetry and Dance
23 Wave Numbers
Truth Factor
Factoring
Aliquot Parts
24 The Quantum Ellipse Breaking Down the Wave Truth Determination
25 Examples
Quantized States
Evaluation
26 Conclusion
28 Tables
46 Iota
47 Detail
The Myriad
The Limit
Quantum Flexibility
48 The First Waterfall
Organizing Energy
Other Waterfalls
49 The System of Myriads Crystal Universe
Unit of Energy
Aggregation
Book 4
51 New Beginnings
Gains Made
Human Senses
52 The Path
Essence of Change
53 Aliquot Parts
Precursor Energy
Secondary Precursors
54 Audible Scale
Visual Octave
Scale Limits
55 Myriad of Mentalism
Standing Waves
Why and Wherefore, The Roots 56 Par Types
Koenig Series
57 Other Texts
Parameters
58 Progress
Energy is Information
59 Parameters
The Message
Sensory Input
60 Information
61 Platonic Solids
Ending
Conclusions
Different
Uses
62 Where have we been Next
New Foundations for Theory 63 Disremembering
Use of These Books
Rebuilding the Future
Previous Books
64 Suggestions
65 Glossary
ILLUSION
Quantum Arithmetic
erties of the halogens, and particularly chlorine. It has been necessary to outlaw the use of carbon
This chapter should have occurred much earli er in these texts in order to clear the air. But hav ing gone this far, perhaps the reader can more benefit from these thoughts by this time.
Surely Quantum Arithmetic is a difficult sub ject, not that it is difficult in itself but because of ingrained thought patterns of the student who is trying to understand. Quantum Arithmetic is NOT the ordinary mathematics which it seems to be. This is the GRAND UNIFIED FIELD and it derives from ancient mathematics.
In the past text, I have refrained from criticiz ing contemporary science and scientific methods because that would be counter productive. But let me review the present status of science:
(1) Every scientifically inclined person is amazed at the progress we have made, and it is phenomenal with our entry into space; Our devel opment of the atomic bomb, and atomic power; With our development of laser and diodes; Our state of the art in computers, and so many other areas. But we have become insolent in thinking this is all there is. How little do we really know? It is about ten percent of what every person should be able to understand.
(2) There is absolutely no area which is not fraught with dangers of accident. We have literally built our scientific institution on a foundation of sand, and even today we can begin to see it crum ble away. Those little things we have ignored have come into primary importance. We have become conceited with our progress.
(3) We have overlooked those small trivialities and made our discoveries only part way. With the Superconducting Super Collider, (SSC) we are like little a boy who receives new electric train set for his birthday. With the superconducting Super Collider, the scientists will be pushing around atomic particles instead of railway cars. They will see magnetic controls for their railroad track. In this case they are using a twenty pound maul to drive a carpet tack. The work they are trying to ac complish can be done more effectively with pre cisely controlled harmonics. Certain few scientists are beginning to do just that today.
(4) In overlooking the trivialities, on which Quantum Arithmetic is based, chemistry alone is capable of destroying the habitability of Earth. It is presently destroying the ozone protective zone in the stratosphere. This occurs only because we have overlooked some of the minor radiative prop
tetrachloride, and trichloro ethylene. Chlorine is a male (radiative) element and it goes out of its way searching for trouble to cause. It’s antigen is sodi um and potassium, the two main counteractive fe male (attractive) agents.
(5) The word "Quantum" is overworked and misused. It has become a hype word surrounded with mysticism. Look in a dictionary for the defini tion of Quantum. No two of them are alike, and each one goes to great ends. The definition was given earlier. Quantum means only: "BASED ON WHOLE NUMBER VALUES." It means neither more or less but it does require a working knowl edge of prime numbers because primeness or co primeness must be maintained in some composite numbers are required.
(6) Everything created by nature is quantum. A particular case is a cogwheel or a gear train. It is based upon the variations found within the prime number system. Each gear wheel can be re duced to its prime number of gear teeth, or the prime factors of number of gear teeth. In getting away from this in our human creations, we have created a system of chaos. There is literally noth ing in human creation, which is quantum. Laser is one major exception and that is because we have, unknowingly, used the quantum numbers of the natural energy states of the atoms. We could do much better in creating crystals if we would better understand the quantum formation of various crystals.
(7) In mathematics we have become far too complex. It takes a lifetime to learn contemporary mathematics and it is all chaos in the end. Quan tum Arithmetic becomes the essence of simplicity and it attains absolute accuracy. The numbers es sentially stop with 1, 2, 3, 5, 6 & 7. (Pythagoras said to 10). The higher prime integers, 11 and 13, come into play in a secondary role. Stopping at 7, we can then continue to 16, and just short of 17, just as Pythagoras claimed. The numbers 2, 3, 5 and 7 will carry us to 16. (16 = 24; 15 = 3 x 5; 14 = 2 x 7; 12 = 3 x 22; 10 = 2 x 5; 9 = 32; and 8 = 23.) These use only the integers up to 7, leaving out 11 and 13 which play only a secondary role.
To support Pythagoras, remember that 1 x 2 x 3 x 4 x 5 x 6 x 7 = 5040. But at the same time 7 x 8 x 9 x 10 = 5040 also. This is truly a magic num ber in Quantum Arithmetic.
After 16, the next stopping place for secondary prime integers is somewhat less than 97. The ter tiary integers, which are all composite integers,
1
Quantum Arithmetic
stop at 10,000, and absolutely no higher integers are needed. This will be explained in the following chapters. This list could go on to area after area in all of science, but let us get into a more positive mode.
We have not recognized that everything in na ture, (but not in human constructions), from elec trons to galaxies conform to absolute quantum laws. From electrons to galaxies includes only matter. But energy is the parent of matter, not the other way around. Energy also has other children. One of them is audible sound and another is visi ble light. And it appears there may be at least five other children which we are yet to discover. Quan tum Arithmetic will lead the way.
Once a person recognizes that energy is all that really exists in nature, we begin to make headway. Everything else is made from energy. Matter is made from positive energy, for protons, negative energy, for electrons, and there are lesser details. All of our current scientific discovery per tains to matter in some way or another, yet matter itself is less than ten percent of the whole energy spectrum. It can be estimated that we have brok en into nature to glean less than five percent of the information which can be available. Most of that is faulty and incomplete in one way or an other.
Beginning with energy, everything in nature is precise, and exact. This is made possible through quantum laws. It is a precision which we can nev er attain unless we use the mechanisms which nature has provided. We have done that, (un knowingly), in the case of laser. Frequencies, must be accurate within millionths of one hertz. In most places we need to produce accuracy, to 0.0001 of one hertz. Our human produced frequencies are not that accurate and therefore human produced things can almost never be quantum. That is the difference between conventional science and quantum science.
With the array of ten thousand quantum val ues between two sequential hertz values there will be a quantum value which nearly matches our produced values. But nearly, is not close enough. This will be explained more fully in the remaining chapters of this volume. It must be absolutely cor rect. That’s what Quantum Arithmetic can deter mine, first through quantizing and secondly by finding the prime factors of the quantum num bers.
For instance, say we are making the gears for a clockworks. We make the teeth on the gears pre cise from beginning to the end of a gear wheel. At
the last tooth we find that we have a part of a space left over between the last tooth and the first because the diameter of the blank wheel was slightly over or under. It is possible to make such a gear operate by applying enough power to force it over this obstacle. It will damage the gear, but it will never truly wear in if both gears have a prime number of teeth. In this case, every tooth on one gear will eventually mate with every tooth on the other gear. That is what quantum is all about.
This seems such a trivial thing. But if we do not consider all such trivialities we cannot, ever achieve quantumness in our constructions. The values must be absolute, and must be discrete. Only integer values can be used.
TO BE PRIME
Considering integer values, some number the ory, (or rather Quantum law), must be considered. When is a number a prime number?
In Quantum Arithmetic, every number is prime to certain others. Every integer is prime to any integer which is one unit less or one more than itself.
Every integer is prime to any integer which is two units more or two units less than itself, pro vided the original number is not divisible by two.
Every integer is prime to any integer which dif fers from itself by a prime number. This can be taken as a corollary to Euclid VII, Proposition 28. In this way, two integers will be prime or co prime, to each other. Euclid expands this to 4 in tegers. Book VII, Proposition 28 states that the sum and the difference between two coprime inte gers will be prime to both of them. This creates the four integer sequence which has been called the “quantum number”. Coprimeness is one of the qualities which is required of quantum numbers.
Most readers lapse into unreality when it is claimed that Quantum Arithmetic is based on un derstandings of the ancients. There are many an cient beliefs that dwell on the trivialities and the philosophical knowledge that they gleaned from them in an analogous way. Plato claimed that life to death to rebirth was an analog to wakefulness, to sleep, to reawakening. This was derived from mathematical attributes.
When the reader reads such passages, it brings visions of mysticism, and the unreal. But is anything which is not readily understandable also mystical or unreal? When a missionary pro duces a mirror to an aborigine, or fire from a
2
Quantum Arithmetic
match, is that mystical? To us, no, but to the abo rigine it is mystical. Is it unreal? No it is real to both us and the aborigine because it can be ob served in occurring in the material realm. It is not illusion. Only our interpretation may be illusion. In the case of Quantum Arithmetic, reality can be demonstrated in pictures and diagrams but math ematics is not in the realm of matter. We can un derstand it only in the sense of abstract logic and not with our normal senses. It is abstract number, but can be used in relation to matter or any other natural thing such as music, forces, or energy. It is not mystic, just because we cannot put our hands on it. We are familiar with numbers, but here they are put in a different context concerning the relationships between specific numbers.
Relationships have been used repeatedly in connection with engineering and design, but only in a way of relative values. In Quantum Arithmetic we can begin to attribute physical relationships in a generalized way. When relationships have been tested against our familiar knowledge and correla tions have been found, it is no longer mystical. It is now metaphysical, which means it is beyond our tangible senses, because it has been discon nected from matter, per se. Our senses pertain only to matter and material things.
Because we can approach Quantum Arithme tic only mentally, does not mean that it is unreal. Having a science that is limited to the realm of matter, being less than ten percent of the energy spectrum, we have learned many things in science which are not entirely true or complete. What in our knowledge which is untrue, must be disre membered. What is incomplete must be complet ed. That’s the crux of our difficulties with Quan tum Arithmetic.
One more thing which can enhance our belief in Quantum Arithmetic: Many of the relationships have been generalized to an extent they create the “exceptions” to every rule. Quantum Arithmetic overcomes the many "exceptions" which science has had imposed upon it. The limitations of sci ence to things material, has previously prevented making these generalizations. Now that the gener alizations can be made we find minor and major errors in definitions within science. Correcting these definitions, as we have corrected the defini tion of quantum numbers, clears the way for fur ther progress.
Here in Book 3 is found a major revision in wave theory. The mathematical background of harmonics, now being known, allows us to make creations without the previously unwanted har monics. Potential harmonics can now be exposed
mathematically. Our designs can now be more precise, more efficient, and more durable.
Attempts have been made to simplify the frame work of writing these texts. The attempt has been made to make these texts readable and un derstandable to the lay public. Discussions have been reduced to a low denominator, even though that creates extra length to the text. Most abbrevi ations have been eliminated to avoid confusion and to improve understanding. It is hoped it will make these accessible to the general mass of readers.
There are those who distrust and fear science on the one hand, and those who think our present science is already completely known on the other. It is hoped that this is accessible to them both. But if their mental processes are so crystallized they cannot absorb the message, so be it. Let sci ence again become a subject in which the general public can participate.
There are many between these two extremes who have hope and a faith in the future for our children. And there are thousands working to build that future. It is hoped these pages will make this accessible to them and to the general mass of readers, and who are interested in new items of scientific knowledge with which to build that future.
This new beginning need not be made with confusion and chaos while we clear away the old misconceptions. The old theories and hypotheses must be reviewed. In some cases they must be abandoned altogether. In most cases they need only revision. Let the original experiments be re performed in this new light, when so indicated, and the new interpretations formulated.
Gradually we should begin to see the new light of day, and the new age will be upon us. The fol lowing chapters will enter what so many have called metaphysics, and others have called mystic areas. They are mystical only to those who do not understand the beginnings laid down in these texts. Those mystical areas have a rationale. We must go beyond the world of matter in order to understand it.
The following chapters will go beyond the world of matter and our perception of matter through our senses. Using the information which was put forth in the previous volume of Quantum Arithmetic, we will deal with its dynamics. The dy namics of numbers, through Synchronous Har monics builds a bridge or understanding energy as we find it in the physical world.
3
Quantum Arithmetic
The present decay and chaos which is occur ring in contemporary science makes it mandatory that we go back to our foundations in mathemat ics and connect those quantum foundations to what we know and can prove correct in current scientific theory. There must be corrections made to those theories which were derived from faulty interpretation of experimental results.
When this is done, we will find that many of the exceptions which presently appear, will dis solve into thin air. Some of these corrections have already been suggested, and some will be de scribed in the following chapters.
Chemistry presents an opportunity to make a constructive change with our new found ability to quantize all energy states of all atoms of any ele ment. This, in turn, teams up with what we know about music or to develop new chemical com pounds and the possibility of knowing beforehand the characteristics of those compounds even be fore they are produced.
Work is in progress for the redevelopment of the musical scale for Music of the Spheres as de scribed in ancient legends and texts. Along with this we find that Lord Rayleigh while contributing to our present wave theory, also contributed some errors of interpretation. Music itself offers a wide field of applications to be derived from better un derstanding of Quantum Arithmetic.
The relationships of Quantum Arithmetic in motion help us to present proposed improvements in understanding wave theory as it applies to us age when separated from matter.
An understanding of harmonics of energy cy cles explains many of the phenomena we have en countered, both synergistically and devastatingly. Had we had this information previously, many major accidents could have been prevented. Infor mation gleaned from accidents could have been put to good use elsewhere.
When we, being composed of matter, consider that matter is of primary concern in the creation, we put ourselves somewhat in the position of "Flatlanders", unable to perceive the third dimen sion. Matter is not a primary parameter of the foundations of science. Matter is a secondary creation, secondary to the vibrations and cycles of energy.
As will be explained in future chapters, energy of vibration is divided into several stages of magni tude and perhaps the Biblical seven stages of creation. Each stage is one magnitude of energy
values, or of hertz, if you please. The production of matter falls in only one of those stages. Three other magnitudes have been located. The highest is that magnitude containing the octave of visible light. Another magnitude is of audible sound ranging from approximately 30 hertz to maximum of ten kilohertz. Below that is still another magni tude, which concerns us directly and is the mag nitude of mentalism.
These three magnitudes are directly observa ble from our magnitude of created matter. It is possible to fathom them through our senses, but our senses are restricted, and limited to this range. There are other magnitudes above and be low this range. Our knowledge of astronomy falls in the category of measurement of magnitudes by its periodic cycles.
So this provides a brief outline of what will be studied in this volume, and a warning of the diffi culties the readers may expect. It will be up to you to judge what is reality and what is illusion. Reali ty means that something really exists whether we can sense it or not. Illusion means that something only appears to exist when in fact it has no place in creation. Lord Rayleigh believed the higher har monics were created by an audible note. That was his illusion. The reality is that the lower note is created as the sum of the harmonics above it. Without them the note could not exist. That is the reality.
TEST
1. In an ellipse, J=28 and K=70. What is L? If your answer is not 140 then go back to Book 1.
2. Same ellipse: How many quantum points are there in this ellipse? If your answer is not 84 then go back to Book 2.
If you answered both questions in less than one minute then congratulations.
If the answer required 5 minutes, you are sat isfactory.
If you took more than 10 minutes, perhaps your arithmetic needs practice.
Synchronous Harmonics leads us in a direc tion to a better understanding of energy. The pre vious chapters of Quantum Arithmetic concerned the STATIC phase of numbers. This chapter en ters, now, into the DYNAMIC phase of Quantum Arithmetic which is called “SYNCHRONOUS HAR MONICS”. This dynamic aspect takes the num bers, in their relationships, as they repeat them-
4
Quantum Arithmetic
selves, in continuing sequence, as trains of waves. These waves of energy progress in step with each other.
As the mathematics of Quantum Arithmetic is placed into motion, through Synchronous Har monics, drawings of various wavelengths in pairs and their composite, explain more about waves. The importance of the 4-way division of integers, becomes apparent, and we also begin to under stand "wave packets" better.
TIME SYNCHRONIZATION
Synchronous Harmonics is the dynamic rela tionship between numbers and number groups. Each number seems to start at one unit and re peatedly builds itself to their full value in a contin uing process. Each number reaches its full value in due course and starts over. When two or more numbers are working together they seem to stay in step unit by unit like cogs on a wheel. One can imagine these number progressions are controlled by synchronized units of time.
Take the two simplest numbers, 2 and 3, to see how they progress. They both start at 1 and 1. At the next step they progress to 2 and 2. The "2" is completed and ready to start over. (See below). At the next step they are 1 and 3 which completes the "3", preparing the "3" to start over, leading to their values as 2 and 1 at the fourth time-jump. At the fifth time-jump they will be 1 and 2, and the sixth time jump they will be 2 and 3. At this point they again come into synchronization as they were, at the end of one cycle and the begin ning of the next cycle.
These can be listed by their quantum time:
Time “2" “3"
0) 0 0
1) 1 1
2) 2-0 2
3) 1 3-0
4) 2-0 1
5) 1 2
6) 2-0 3-0
The “2" goes through cycles and the “3" goes through cycles. 2x3=6. All through their progres sion they have different (instantaneous), compara tive value. At each step, they never repeat a rela tionship in values within a cycle so long as they are coprime.
ENERGY IN MATTER
These must be considered as energy values which are completely divorced from matter. Mat ter represents a static relationship between num ber groups as time progresses. Energy progresses through this dynamic process within mathemat ics.
Matter is static in nature. It is theoretically composed of precise frequencies of energy to form a standing wave we call matter. Once formed, matter can absorb and release surplus quantities of certain frequencies. The physical characteris tics of matter result from the alteration of energy content of that matter.
Matter can change in its heat content but heat is energy. The change is in the amount of energy bound up in the matter with no permanent change in the matter. Its absorbed energy, or tem perature, is only a transient change. Heat energy can also cause matter to expand or shrink. It is this energy which is the subject of Synchronous Harmonics.
The study of energy has always been pursued by considering energy to be composed of waves. The values, as given above, could be the instanta neous value of each wave in relationship to the other. The “2" goes through three cycles while the “3" goes through two cycles. This is because 2 and 3 are PRIME to each other. They both coincide at their product 6.
From these examples we can guess that the correlation between 5 and 6 might be. They would coincide at 30 which is their product. From zero to thirty, they could form every combination of 0-5 and 0-6 without repeating any combination, until they culminate at 30:
Time “5" “6"
1 1 1
2 2 2
3 3 3
4 4 4
5 5-0 5
6 1 6-0
7 2 1
12 2 6-0
13 3 1
14 4 2
15 5-0 3
16 1 4
17 2 5
25 5-0 1
5
Quantum Arithmetic
26 1 2
27 2 3
28 3 4
29 4 5
30 5-0 0-0
The “30" is considered the ending, or synchro nous point, but it is not the harmonic point. The harmonic point will be described later.
Let us consider how 5 and 7 will correlate. The 5 must go through as many cycles of its value as the value of the other number, (7). The first num ber is 5, so it must go through its instantaneous values of
0, 1, 2, 3, 4, 5-0, 1, 2, 3... seven times.
The other number, 7, must go through 5 cy cles of 0, 1, 2, 3, 4, 5, 6, 7-0, 1, 2, 3, 4,... . The 5 cycles of 7, and the 7 cycles of 5 will coincide, (synchronize), at 35 because 5 x 7 = 35. The table will look like this.
Time "5" “7" /// Time “5" “7" 0 0 0 || 19 4 5 1 1 1 || 20 5-0 6 2 2 2 || 21 1 7-0 3 2 3 || 22 2 1 4 4 4 || 23 3 2 5 5-0 5 || 24 4 3 6 1 6 || 25 5-0 4 7 2 7-0 || 26 1 5 8 3 1 || 27 2 6 9 4 2 || 28 3 7-0 10 5-0 3 || 29 4 1 11 1 4 || 30 5-0 2 12 2 5 || 31 1 3 13 3 6 || 32 2 4
14 4 7-0 || 33 3 5 15 5-0 1 || 34 4 5 16 1 2 || 35 5-0 7-0 17 2 3 || Product and 18 3 4 || Synchronize
They will coincide for the first time at 35 which is the product of 5 and 7. They will be shown later as sine waves.
EVEN NUMBERS
What happens when two numbers
chosen are 4 and 6 which are not prime to each other? The “4" is 22, and the "6" is 2X3. Altogether there are three 2's in the factors, and only one 3. The results are given in the following table. The high er power of 2 will remain but the lower power will
be absorbed in the higher power. Synchronization
of 4 x 6:
Time
First
Seco
nd
0
0
0
1
1
1
2
2
2
3
3
3
4
4-0
4
5
1
5
6
2
6-0
7
3
1
8
4-0
2
9
1
3
6
Quantum Arithmetic
10 2 4
11 3 5
12 4-0 --- 6-0
In this case they do not coincide at their
product, (4x6=24), because they have the com
mon factor of “2". They coincide at half of their
product because they are not coprime. (They act as though the two numbers were "3" and "4"). At each step along the way the "4" and the "6" have different relative values without repeating that rel ative value. They proceed through only half of the expected combination of values.
GRAPHICS
The waves of 3-units, 5-units, 7-units and of 9-units are shown in various combinations. The first two diagrams show waves of the same par value. The first graph, the pair of waves, (3 & 7) 9 are both 3-par. In the second graph, (5 & 9), are both 5-par.
The shaded areas at their 1/4 points, show the two points at which each pair of waves will coin cide at their maximum values. The value of the wave, (that is height of the composite wave), is greatest at the one-quarter and the three-quarter point of the combined cycle. These are the HAR MONIC points of each combination. When both waves of a pair are of the same par-value they will reinforce each other.
This is the mathematical equivalent of the type of "wave packet", in which two different waves will reinforce each other.
The 5 & 9 waves coincide at 45. Only the first half-wave is given. The second half is a mirror im age of the first.
In the case of the 3-unit wave and the 7-unit wave they will HARMONIZE at 1/4 and 3/4 of their product. Since the product is 21 the surges will occur at 5 and 1/4 units, and at 15 and 3/4 units.
In the case of the 5-unit wave and the 9-unit wave, their product is 5 x 9 = 45, so the maximum point of the surge will occur at 11 and 1/4 units,
(as shown), and at 33 and 3/4 units.
Only the first half of the 5-units and 9-unit waves is shown. If the first quarter of this wave, as shown is rotated about the vertical axis at 11 and 1/4 units, it will complete the first symmetry. Then if all of this wave is folded at the vertical axis at 22 and 1/2 units and the last half rotated about the horizontal axis, it will complete the wave at 45 units. The last half of the wave is symmetrical to the first half. Also, each quarter of the wave is symmetrical with all other quarters. It is symmetrical in exactly the same way the recip rocal of 17 is symmetrical in problem No. 9 at the end of this chapter. The difference is that in prob lem No. 9 the work is STATIC, but here it is DY NAMIC.
When one wave is 3-par and the other is 5-par, the composite graph will be changed considerably.
The 3 and 5 waves will oppose each other at 1/4, and at 3/4 of 15, because one is 3-par and the other is 5-par.
The 5-unit wave and 7-unit wave, will oppose each other at 1/4 and at 3/4 of their product, 35. At these points they will essentially cancel each other forming the second type of wave packet, which is sometimes called the "null wave packet". (Only the first half of this graph is shown. The other half is a reflection).
When two waves are of the same par type they will support each other at the quar ter points. When they are of opposite par type they will oppose. The net result is that the com posite of the first graph is much smoother. The second type tends to have more spikes and sharp er points. This latter graph, of 3-par and 5-par waves, will also have flattened spots at the quar ter points.
EUCLID'S 4 NUMBER TYPES
This demonstration through Syn chronous Harmonics, more clearly shows the rea son for considering the division of integers into the four par types. These four par types are:
7
Quantum Arithmetic
wave may begin. It will be in phase and eventually
2-par, (even-odd, 4n-2); 3-par, (odd-even, 4n 1); 4-par, (even-even, 4n); and 5-par, (odd-odd, 4n+1), waves.
This is missing from contemporary mathemat ics. Contemporary mathematics acknowledged the 4n+1 and 4n-1 integers, but completely missed the full gravity of them. It also completely omitted the differentiation between 2-par and 4-par inte gers.
WAVE PACKETS
The four-way division of integers, as Euclid named them, leads to the formation of Wave Pack ets. But only the 3-par and 5-par waves contrib ute, and they contribute only at the harmonic point of their composites. At quarter points of their product, when several 3-par waves, or wave lets, harmonize, they will form a greatly enhanced composite wave. Ovid. W. Eshbach, "Handbook of Engineering Fundamentals", John Wiley & Sons, (1952) defines wave packets: "- complex, yet, peri odic wave forms can be obtained by an algebraic superposition of several waves of different parame ters. The superposition of waves with infinitesi mally different frequencies will, at a certain time, t, give an absolute maximum for a point x. They progress at any other instant to give a maximum, periodically.
Only the 3-par waves, or the 5-par waves will superimpose to enhance a wave packet, at their harmonic points. The 3-par and 5-par together, when equally balanced, will form a "null" packet. (This is demonstrated in the next chapter.)
The composite values of these various graphs demonstrate the part this 4-way division of inte gers play in creating the spiked wave packet and the null wave packet.
These will prove, eventually, to be very impor tant to future wave theory developments, and will enable us to reach knowledge which has been un available to us because it is completely beyond our sensory capability. These will be followed up later in this chapter. This also demonstrates the reason Euclid claims there is a 4-way division of integers into even-even, even-odd, odd-even and odd-odd. It explains that which Sir Thomas Heath puzzled over in Book VII of Euclid.
PHASING IN
Any odd-valued wave will automatically phase in although this is not readily obvious. It makes no difference on which integer any odd valued
reach a synchronous point with all other odd val ued waves. This occurs for exactly the same rea son that the remainders of a division process, must go through all integers which are less than the divisor when prime numbers are involved. (See Problem 9, page 9.)
No odd-valued wave will approach the baseline from below at an integer. It is always at a half integer. When two waves are not coprime, they will come into phase at their product, after com mon factors are removed. If any of the prime num bers also have higher powers, only the one high est power is retained, and lower powers are dropped. This reduces the factors to coprime stat us.
Contemporary mathematics is unconcerned with factoring wave values into prime factors. In deed, this could not be done because quantum values are not used, and could not be known until now.
All waves with which we are familiar are com posed of not more than eight prime factors. How ever, they may contain as few as four prime num bers for the male waves and possibly as few as three prime factors for female waves. The male wave, such as the one with a quantum number like 17, 32, 49, 81 will have only four prime num bers 17, 2, 7 & 3, or their powers. The female wave, such as 2, 7, 9, 16 will have only three prime factors, 2, 7, & 3. Other such waves will also exist if the difference, between two powers of 2, is equal to the sum of two prime numbers. These are rather unique sets, and it is thought they are quite rare. In the case of the female waves, they must begin with 2 because any other 2-par integer will include a second prime number. They may not begin with a 4-par number and end with a 2-par integer. Whatever the case, no male wave can have less than four prime numbers, and will most generally have seven factors.
On the other hand, the coming into phase is an essential part of all harmonics. If it were not for this automatic phasing-in, music would find it impossible to form a musical chord. Some of the following problems partially demonstrate this phasing in process.
What is it about Quantum Arithmetic that makes it so difficult to understand? There really are no new basic facts that were not known in conventional mathematics. What Quantum Arith metic has done, is to take many of the trivialities which were known but were passed over in con ventional mathematics. What is new is in showing
8
Quantum Arithmetic
the relationships between these trivialities and
showing how these relationships are all tied to gether and form the foundation of conventional mathematics. But the missing mathematical foun dation of our sciences is only found in Quantum Arithmetic.
The mathematics of Quantum Arithmetic is so interlocking and so simplified that it becomes de ceptive in the ends which can be reached. One is tempted to challenge its existence but one cannot challenge the mathematical proofs on which Quantum Arithmetic is founded.
Through Synchronous Harmonics we may be able to understand why Sympathetic Vibration oc curs. On this approach through static Quantum Arithmetic and its dynamic stage, Synchronous Harmonics, we are better able to see the immense system which the prime numbers give us for un derstanding science and nature. Quantum Arith metic is not Number Theory, but it does extend our understanding of the number system.
Harmonics, alone, have created numerous problems in present technology. A better under standing of harmonics, and the true parameters which must be considered, can help prevent those accidents.
It is a long road. First we must be able to pin point the true parameters in any construction, or creation. Then we must be able to Quantize those parameters into terms acceptable to Quantum Arithmetic. In this way we can determine the help ful, and the harmful, harmonics which are intrin sic in any proposed construction. We can be able to determine the true harmonic analysis in any design, and the probable harmonic stresses which that construction must withstand. This applies to earthquake hazard as well as all other energies which our designs must resist or use.
What has been discussed in this chapter con cerned only waves in pairs. A previous paragraph above stated, in effect, that no wave exists which has only two prime factors. There is only one ex ception, and that is the first, based upon the quantum number 1, 1, 2, 3, which has less than three prime factors. It has only the prime factors of 2 & 3. This generates the "unity" prime right tri angle -- the 4, 3, 5, basic triangle which will divide into every larger triangle, and represents the "IOTA" which will be introduced much later.
The following chapter will begin to discuss the empirical waves which we will encounter which have from three to seven prime factors.
PROBLEMS AND QUESTIONS:
1. Sketch waves for the correlation between a period of 5 units and a period of 11 units.
(Use half circles for simplicity.)
2. Sketch waves for the correlation between a period of 7 units and 11 units. Start them at any unit on the graph paper. Where they synchronize will automatically be 77 or zero, as they phase in.
3. Sketch waves for the correlation between a period of 5 units and 8 units.
4. A bicycle has 37 teeth on the pedal sprocket and 13 teeth on the rear wheel. How many times must the pedal sprocket turn to again be in the same relation to each other?
Ans: 13
5. In the above problem, how many revolutions has the wheel made:
Ans: 37
6. Both sprockets have circulated the same number of teeth in this cycle. How many teeth passed a given point on each before the gears return to the original relationship?
Ans: 13 x 37 = 481 teeth.
7. If the bicycle chain had 104 links, How many links must pass before a given tooth on the pedal sprocket re-engaged any given link a second time?
Ans: 3848. (because 13 and 104 are not coprime and 37 x 104 = 3848)
8. In the above question, how many links must pass a given point before a given tooth on the rear wheel sprocket re-engages a giv en link for the second time?
Ans: 104 (Because 13 x 8 = 104). Any given tooth on the rear sprocket would en gage only eight different links. It would never en gage the other 96 links.
9. With pencil and paper, divide 17 into 23 until it repeats. List each digit of the quo tient and below it list the remainder for the next division.
Ans: The division must go through
9
Quantum Arithmetic
every remainder less than 17 before it will repeat because both numbers are prime. The division will give:
1/6, (3/9, 5/5,
2/16, 9/7, 4/2,
1/3, 1/13, 7/11)
( 6/8, 4/12, 7/1,
0/10, 5/15,
8/14, 8/4, 2/14),
(repeat), 3/9,
5/5
10. Divide
37 into 104 as
in problem #7.
It should go
through 36 re
mainders since
37 is a prime
number, but it
does not. It re
peats after
three digits.
Why? This oc
curs because 37
divides evenly
into 999, and
the primes, 2 &
5 enter the pic
ture through
the decimal sys
tem.
MULTIPLE
WAVES
With the simple combinations behind us, this chapter will look into what is beyond. It appears that no wave can be formed with less than four prime, or coprime, integers. There must be the 2, 3, 5 and/or 7 along with one larger prime num ber. Go back to the two waves plotted on a single straight line. Instead of a straight line, let us plot them on a line that closes on itself in a circular or elliptical line. This closed line will represent the straight line used previously.
THE HARMONIC CYCLE
A drawing of the plotting of a 3, a 4, and a 5- unit cycle on the circular line is given. In this case, the closed circular line is 60 units in cir cumference. This can be considered as one "Har monic Cycle" which is being theorized. In this case the larger circle closes at 60 units. The three plot ted cycles, 3, 4, and 5 also close at the same point
-- 3 x 4 x 5 = 60 unit cycle.
The phase relationship between the 3, 4, and 5 waves repeat themselves only at 60 units. But
the 3 and 4 re
peat their phas
ing, 5 times, (at
12, 24, 36, 48,
& 60); The 3
and 5 repeat
their phasing 4
times, (at 15,
30, 45, & 60);
And the 4 and 5
repeat their
phase cycles 3
times, (at 20, 40
& 60).
Connecting these in-phase, (Syn chronous), points for each pair will inscribe a pen tagon, a square, and an equilateral triangle re spectively. These happen to be the three plane shapes which compose the five different Platonic solids. The application of this configuration will be discussed in a later chapter in connection with the first creative "Myriad"; And in connection with the formation of "standing waves". This formation is one of the features which can promote the "wa terfalls", which are also discussed in a later chap ter.
The 60-unit harmonic cycle is only one of several types. These begin with the circum ference of the larger cycle being 30-units, 42 units or 105 units, and powers of 2, 3, 5 & 7 as multi ples. The 30-unit cycle is composed of 2, 3 & 5 waves. The 42-unit cycle is composed of 29, 39 & 7 and the 105 unit cycle is composed of 2, 3, 5 & 7 unit cycles. The larger cycle on which the small er ones are plotted can be any integer value. The larger, baseline circle, probably can be any value
10
Quantum Arithmetic
larger than 10 units in circumference so long as that larger value does not have a 2, 3, 5 or 7 as one of its factors.
The circular baseline is intersected at all odd integers except 1 and the prime numbers from 7 through 59. These prime numbers are shown by the rayed center showing the symmetry of the twin primes, 11-13, 17-19, 29-31, 41-43, 47-49. Add to this the 59-61. The "3" and 5 are not rayed because they are used in the cycles. So, neither are their supplements 57 & 55. The 61 is in the next turn of the 60-unit cycle. These become sym metrical in 1-59, 7-53, 11-49, 13-47, 17-43, 19- 41, 23-37 & 29-30, the total of each pair being 60. Some of these project to the circular baseline to points at which the 3-wave and the 5-wave oppose each other.
NOTE that 49 is shown as a prime number. That is because it is a power of a single prime number, (7), and that is its only factor. Any power of a prime number is also considered as prime, and will cancel any lower power of that prime root.
Since many quantum numbers contain the prime numbers 2, 3 and 5, this cycle is a part of every conceivable wave.
OTHER CYCLES
What happens if the larger circle were, say, 61 units in circumference? Now the 3, 4, and 5-unit cycles would be one unit short of closing. In this case, we would have to show the larger cycle as a helix instead of a closed circle. Going around the helix again they would be 2 units short of closing. They would have to go around this 61-unit cycle 60 times in order to close at the same time the 61- unit cycle closes. One can begin to see the analogy between this and "the two gears and the chain" in the previous bicycle problem.
The helix would eventually close on itself at 3660 units. Because it closes on itself it is possi ble that the figure would more resemble a lissajou figure, (to be discussed later), rather than a true helix. Every harmonic cycle must contain a 2, 3 and a 5 and/or a 7 (or some power of them). One harmonic cycle, as described, must be a part of any wave that combines with another in a harmo nious way. The composite wave of these low prime numbers constitute the "teeth" which must mesh with another cycle.
PHASING IN (again)
But now suppose that just one of the several waves which compose a harmonic cycle begins out
of phase, or begins on an integer other than zero. This appears to be a situation which would be im possible to manage, but it is not. As described in the previous chapter, there are no two, odd valued cycles, which will approach the base line from opposite sides at the same point. That is to say, no odd valued cycle will be at full cycle when another odd valued cycle is at half cycle.
If cycles are plotted on Harmonic Cycle, start ing at any integer, they would have the effect of simply rotating the zero point on that cycle. Prob lem #2 of the previous chapter helps to under stand why any odd-valued cycle will phase in, re gardless of which integer it begins its cycle. Each odd prime wave must necessarily pass through every possible remainder which is less than itself, a determined in the previous chapter. When in conjunction with another odd, prime-value cycle, the synchronization of these, (their product), will eventually coincide with every remainder of any third cycle.
There is one more condition which prevents any two cycle from approaching the baseline from opposite sides. To approach the baseline, an odd cycle must meet the baseline at a half integer on its half-cycle, as it approaches from the upper side.
To approach the baseline from the lower side the odd-value cycle must be completed and it will strike an integer. That is why two odd cycles can never approach the baseline on opposite sides, at the same point, thereby cancelling each other.
Surely, this sounds very trivial, but it eventu
ally has high impact on the outcome of our devel oping wave theory.
EXCEPTION
The above is not true, in the case of the even valued cycles. A 4-par cycle will always approach the baseline from either side, at an even number. The 2 par cycles will approach the baseline from below at an even integer at the completion of it cy cle. But it will approach the baseline from above
11
Quantum Arithmetic
at an odd integer, at its half-value. At this half cy
cle it can oppose an odd cycle. It is a reason to call the 2-par number "even-odd".
There can be only one, 2 or power of 2, factor in any quantum number. The 2-par waves are the 2, 6, 10, 14, 18, 22, 26 etc. values. Since there must be a 2, 3, 5 and/or 7 in every harmonic wave, the 2-par values below 21, cannot be in any wave where the 2-par value exists with its odd prime factor being 2, 3, 5 or 7. So, the 22-unit cy cle, 2 x 11), is the first such wave that needs to be considered. It will be considered, only when the 11-unit wave, or any 2-par or 4-par wave factor is not present in a quantum number.
Such a 22-unit wave will eventually meet (syn
chronize with) every other prime-valued wave from the opposing side. It essentially absorbs and can
The classical yin and yang makes its appearance here.
It appears between the 2-unit and 4-unit cycles and between the 4-unit and 8-unit cycles. The 6-unit cycle produces a distorted yin and yang in combination with the 2-unit cycle. The braided appearance comes from the reversal points where waves meet from opposite sides of the baseline. What application, if any, it may have, is unknown at this time. It does apply to the male - female division. In this division, the male has the outgoing, radiative characteristic, and the female has the attracting, absorbing characteris tic. It is only part of the reason that most 2-par integers are considered to have the female charac teristic.
MOST BASIC MULTIPLE CYCLE
The following diagrams show the combination of 2, 3, and 5-unit harmonic cycle plotted on a straight baseline. The first is shown in sinusoidal form and in the second the waves are shown as half-circles. These drawings would be bent to join the two ends, in order to form the most basic harmonic cycle. In this particular case the drawing would be 30 units in length. In most physical cases, the harmonic cycle will be covered up in waves which are more complex than these as shown below.
cels a part of the prime valued wave at those points. It is for this reason of absorption, that the 2-par numbers are considered the female, or at tracting and absorbing numbers. There can, and must, be only one 2-par, or 4-par valued cycle among any of those which serves as a baseline, or harmonic cycle.
A drawing of even-valued waves is shown. This composite includes the 2, 4, 6 & 8-unit cycles. They present a braided appearance in contrast to a combination of the odd valued waves.
COMPLEX CYCLES
Most physically applied cycles are far more complex than those described above. The usual wave of nature will have seven prime factors or be made of seven prime waves.
This diagram, is not that of an harmonic cycle because it does not contain a 2 factor. It shows the cycles of the eight prime num bers from 5 through 23, including "9".
12
SYNCHRONOUS POINTS
Quantum Arithmetic
more prime numbers each. There are but eight quantum numbers which contain only the prime
The graph is shown along 70 units of base line. The complete graph would be more than 300 million units in length. The only synchronous points, up to 70, are at 35 & 70, for 5x7; at 45, (5x9); at 55, (5x11); at 63 (7x9); and at 65 (5x13).
HARMONIC POINTS
There are numerous harmonic points which are shown as the shaded areas, at quarter points of products. Of particular note is the area below the line near 33 units. The 5 and 9 are concentric around 33.75, and the 7 and 19 are concentric around 33.25. The first two are 5-par and the 7 and 19 are both 3-par. This is as close as they could possibly be for 3-par wavelet and 5-par wavelet to support each other.
WAVE PACKETS
From about 40 to 50 there will be a flat area in the composite graph, because of the cancellations. This will create a null wave packet. From about 22 to 28 will be an enhanced, (spiked) wave packet in the composite graph because most of the waves are above the line.
It would be difficult to perform a wave analy sis, of waves which are this complex through con ventional methods. With Quantum Arithmetic the precision of the analysis stands out. One must first start with proper parameters for analysis. Quantum Arithmetic does give the precise values to be used for designing a wave.
It will be many years before anyone will use waves of this complexity. Each wave must have a 2, 3, and 5 and/or 7. (This drawing has no even valued wave for "2", but it does have a “9" repre senting the "3". It serves only as a demonstration. It does put us in a position for improving on our wave theory and theory of energy. The following chapters will work toward that end.
ALIQUOT PARTS
What is an aliquot part? An aliquot part is al ways a composite number. It is usually the prod uct of four to six prime numbers which derive from a quantum number.
A quantum number consists of four integers in Fibonacci configuration. These four integers are coprime, but not necessarily being all prime num bers. Usually one of them is prime and the other three quantum integers are products of two or
numbers of "7" or less. (See Problem 8 on Page 9, Book 1.) All other quantum numbers have prime numbers larger than 7 and usually have 5 or more prime numbers within them. There are very few quantum numbers which will have eight prime numbers represented. If there are more than seven prime factors to a quantum number it usually means it is misquantized. So we can safe ly say that each wave derived from a quantum number will have 5, 6 or 7 prime numbers.
An aliquot part of such a wave will be the product of all prime numbers, excepting one of them, in its quantum number. When two different wavelengths have a series of the same prime factors, the product of those prime num bers forms an aliquot part of each wavelength. These aliquot parts will be identical in the two waves except for the carrier wave of the one prime number which is unique to each wave. The full cy cle of each aliquot part may be as pictured on page 12. The difference would be that most ali quot parts of a waves are composed of five prime internal cycles, rather than the three as pictured. They will be much more complex than these two prime wave cycles.
EXAMPLE
The aliquot parts of two waves, X & Y, are 2 x 3 x 5 x 13 x 47 units. These are the prime numbers which are common to both waves. But wave X has 53, of these quantum units in its quantum number. Wave Y has 7 aliquot parts in its prime number. After wave X goes through its wave 7 times it will be equal to wave Y going through its cycle 53 times. The aliquot part acts as a single unit in each case. The aliquot parts be tween these two waves may be visualized as act ing as gears on two different cog wheels. Each tooth would represent an aliquot part. These teeth would mesh perfectly between the two gears. It represents the bonding between two elements or bonding between two musical tones.
Either one of these waves can be divided into aliquot parts in different ways by leaving out a different prime factor. It could then bond to still a third element or tone which had this new aliquot part.
QUESTIONS:
1. How many ways could aliquot parts form in a wave with the quantum number 10, 11, 219 & 33?
13
Quantum Arithmetic
Ans. The factors are 2, 3, 5, 7, 11 & 32. The 32 supersedes the 2 so the factors are 32, 3, 59, 7 & 11. Leaving out one prime factor each time there should be five different ways. However the product which leaves out the 32 and the factors which leave out the 3 factor will be invalid be cause every aliquot part must contain the factors 2, 3 and a 5 and/or a 7. The aliquot parts are: (1) 3x5x7x32 = 3360; (2) 3x5x11x32 = 5280; & (3) 3X7x11x32 = 7392. The invalid ones are: 3X5X7x11 = 1155, & 5x7x11x32 = 12320.
2. The product of 2 x 3 x 5 = 30. List the primes to 30 leaving out the 2, 3 & 5. Below these list the primes from 30 back to 1.
Ans: 1, 7, 11, 13, 17, 19, 23, 29.
29, 23, 19, 17, 13, 11, 7, 1.
Note the symmetry, and the sum of each pair being 30. Notice also that 52 + 5 = 30; That 33 + 3 = 30: And that 25 - 2 = 30. This is a part of the configuration of prime numbers.
3. Do the products of other sets of prime num bers also make up this type of configuration?
Ans: Yes! there are a few. One was 3 x 4 x 5 in the Harmonic Cycle which was pictured and there are others. Find some others. As the prod ucts become larger certain anomalies begin to oc cur. One of the factors must always be a 6 or mul tiple of 6.
HARMONICS
Harmonics is one of the more important devel opments of Quantum Arithmetic. It is quite differ ent from the harmonics we are familiar with. From the background of the geometry of numbers, pre sented in Books 1 & 2, and more particularly the understanding of Synchronous Harmonics, the meaning and impact of HARMONICS takes its place in future development in science and tech nology. With HARMONICS, music enters the field as an exact science which can function with our technology.
Up to this time science has had to operate on the theories proposed by Lord Rayleigh. Despite his extensive and complicated mathematical deri vations, there are gross errors in current theories, because of errors of interpretations within that now outdated mathematics.
That a note produces all higher harmonics was obvious to our senses and in conventional
mathematics BUT it was NOT true. It is only an il lusion. Quantum Arithmetic, and particularly Synchronous Harmonics gives us an idea of how harmonics works. Those higher harmonics of the lower note are there BEFORE the lower note is produced. The higher harmonics working together create the lower note. The lower note does not create the higher harmonics. And that completely turns the tables on harmonics technologies.
The higher harmonics creating lower energy frequencies is what creates entropy. The travel of energy from higher frequencies to lower frequencies is irreversible. We can, however, derive the higher harmonic by cutting into the string of propagation and bleed off this higher fre quency before it fully creates the lower harmonic. This does not negate entropy. It is always done at a cost and results in a severe loss of mechanical efficiency.
Harmonics relates to the relation ship between two or more different frequencies. Finding the harmonics between two frequencies, they must first be quantized individually, to deter mine their quantum numbers and then together to determine their quantum relationship. The next step is to factor each set of four integers into their prime factors and powers of any prime numbers.
The reason they must be quan tized is that values in conventional units of meas ure cannot be factored satisfactorily. For instance a note of 440 A, will factor into 8, 5 and 11 (8x5x11=440). The frequency for C at 263.2 will harmonize with it. Its prime numbers are 8, 7 and 47, (8x7x47/10 =263.2). But it is an absolute re quirement that every number have 2 and 3 as fac tors. They should not harmonize under conven tional mathematical evaluation by factoring but hearing these two notes in a chord we know they WILL harmonize.
So we must quantize those two notes together and come out with numbers in their quantum relationship. Their values should be shown in the ratio of 329:550 instead of 263.2:440. But this is not the 3:5 ratio (which will be discussed below). The factors of 329 are 7 & 47. The factors of 550 are 2, 25 & 11. But quan tizing between them (sum and difference) they have the factors of 2, 3, 7, 13, 17 & 47 indicating some harmony between them. That is the harmo ny we sense. They are in the ratio of 1:1.6717325.
In the 3:5 ratio, (1:1.666667), they should be frequencies of 264:440 instead of 263.3:440 with superb harmony. The notes we use have the factors of 3 & 47 for one and 2, 5 &
14
Quantum Arithmetic
11 for the other, which indicates a makeshift har mony but not the perfect harmony which could be achieved. It is common practice to tune a piano slightly off from the ideal harmony in order to stretch or shrink the scale in order to make per fect octaves.
Octaves in music as a science, are not perfect octaves, but they will produce perfect music, in the frequencies and harmonies of nature. We have developed the Pythagorean scale, the Just scale, the Equal Tempered scale, the Chromatic scale, and so many more, because of our attempts to look at music in an octave-by-octave approach. This is where we leave music as an art, to make it music as a science.
The perfect harmonies in music as a science carry over into other areas. The most apparent area is in the harmonies of colors of visible light. From here it carries over into the harmony be tween certain of the various energy states of at oms of the elements. This brings us into the har mony between types of matter, and music of the stars, and planets. It also carries us into harmo nies completely divorced from matter.
THE MATHEMATICS
In looking at frequencies from a mathematical view, it is found that the quantum aspects show harmony at frequencies having prime factors of 2, 3 and 5 and/or 7 along with two to four other higher prime numbers. Such a frequency will har monize with any other frequency which also has the same factors plus one unique higher prime factor. None of these higher prime numbers may be greater than 100. This requirement applies not only to music, but to any other range such as visi ble light and atomic energy states.
But the reader may ask, "How can we calcu late sound frequencies in the same range of fre quencies, as light, or as heat, or astronomy? It is because we are discussing quantum frequencies, measured in quantum units of measure. For any waves with which we are concerned, the wave length is the product of all prime numbers of which it is composed. The wavelength is the prod uct of the four integers of its quantum number.
That is only the beginning of the answer. The rest will come in later chapters. But to look ahead briefly, nature limits its number system to the range of zero to ten thousand, just as the Greeks once determined. It is thought they actually limit ed it to 5040, because 1x2x3x4x5x6x7=5040. But 7x8x9x10=5040 also. That is one of the reasons Pythagoras claimed that the working numbers
stopped at 10.
Nature has "jumps" in scale, just as we do in measuring distances in meters. When the meters become too great we revert to kilome ters. Then we adopt even greater unit distances when kilometers become too many. Nature does this also. It is suspected it adopts the multiple of 5040 but the value of 10,000 will be adopted here.
When there is a jump in quantum measurements, the original mathematical laws are also recycled to the new scale and will apply just as they did before. It is then possible to ex trapolate from the range of sound to the range of visible light and to other ranges. This applies to harmonics in all ranges, as will be discussed in later chapters. The chapter on Music and the chapter on Chemistry will describe specific ex trapolations.
INTERDEPENDENCE
There is an interdependence be tween Synchronous Harmonics and simple Har monics. In saying above, that harmony depends on similar sets of prime numbers in different waves. The product of these like prime numbers is what is referred to as an "aliquot part". For har mony, the aliquot part must have primes of 2, 3 and 5 and/or 7.
WAVE PACKETS
Studying the 3-par cycles opposing the 5-par cycles explains the ultimate cause of the peaked wave packets and the null packets. They always occur at harmonic points between two or more waves. These waves, 3-par and 5-par wave lengths, will tell us how prime numbers work to gether. The cause goes back to the definition of the 4-way division of integers, and particularly the 3-par and 5-par cycles. The 3-par and 5-par cy cles concern the male/female division of numbers, which in this case are normally defined as right and left polarities.
PHASING (again)
Understanding why any given cy cle can begin at any unit on the graph, is impor tant to harmonics. Starting a cycle at any unit causes a new LAW of harmonics to be invoked. This causes the whole graph to change to a new synchronous point, with its new harmonic points of the remaining cycles. This helps us understand how nature works in very small quantum parame ters. Considering this through logic we can then see how any musical note can always harmonize
15
Quantum Arithmetic
with another in the same way. If it were otherwise, nature could not operate. It also tends to empha size that there has to be a quantum unit of time to accomplish this feature.
EXAMPLES
There are physical examples of harmonics, two of which come to mind from a century ago:
PHYSICAL HARMONICS
In the 1800's, majestic pipe organs were made which produced sounds down to 16 hertz, (vibra tions per second.) To produce such a low sound would require a pipe more than sixty feet in length. The makers of the organs found they could produce these low sounds with two small, peanut sized pipes. These two small pipes would cascade their ultrasound to longer audible tones. They each produced sounds which were too high to be audible, but together they produced those notes as low as 16 hertz. This will be referred to as a "Cascade". The tuning of such pipes was a trial and error business. The high pitch produced the lower tone and not the reverse as claimed by Ray leigh. We can now mathematically design ultra sounds which will cascade to a precise lower fre quency.
Recently these ultra-sounds were produced electronically with known wavelengths. They will produce any tone in the audible range and higher. In order to do this the two inaudible tones must be in a ratio of a low fraction, (halves to sevenths) to each other. Of course, if the second tone is 8/7 of the first, then the first must be 7/8 of the sec ond. This may help us to understand harmonics a little better. This fractional ratio can be carried to 11ths and 13ths, but the harmony between them decreases substantially.
A second case occurred in 1873-74 when John Tyndall was testing sounds for foghorn use along the English Coast. He found that low sounds usu ally projected much farther than high pitched sounds. But there were unexplained periods when the high pitched sounds would travel 18 miles while low pitched sounds could barely be heard at 3 miles. That is the primary reason that fog horns at many light houses sounded two tones. This paradox is still unexplained but it has been con firmed. [John Tyndall, "The Science of Sound", Cit adel Press (1964)].
Certain machines will produce high pitched sounds and low pitched sounds but do not pro duce many pitches in between. One of these ma chines is a jet engine. It was found, in listening to
planes flying overhead, that high pitched sounds could not ordinarily be heard. But there were short periods of ten minutes to several hours when the high pitched sounds were more percep tible, and the low pitched sounds were negligible, still, no explanation has been found.
A more recent case of catastrophic harmonics was in the collapse of the Puget Sound Tacoma Narrows Suspension Bridge, 50 years ago. Harmonics was the cause, but the science of harmonic is not sufficiently known to prevent such a thing from happening again. All that can be said, for sure, is that the wind velocity was feeding high pitched energy into this bridge, at cascading harmonic wavelengths which the struc ture would accept until it was more than this bridge was designed to withstand.
It is exactly the same case which requires a column of marchers to break step while marching across a bridge. In both cases there is a cascading of energy which is made possible by the "Quantum Flexibility" discussed later. In the case of the bridge, several 5-ton weights placed at prime intervals along the bridge, to break the har monics, could have avoided this failure.
Another case where harmonics are concerned, is in our system of electrical distribu tion. Certain limited cases have been diagnosed and avoided. But there are still major problems which can create havoc. Such a case occurred in the major blackout of New York state, in the 1960's. This was quite possibly a problem in not understanding harmonics.
It is difficult to describe how Har monics works because it is continually moving. It is dynamic. It is like trying to describe all of the waves of the Sun's rays, interacting with each oth er, and continually changing.
THEORY
In working toward a more proper derivation of a workable wave theory, certain ex perimentation was required. Some of the exam ples will be given, along with an improved conclu sion on theories.
Up to this point, wave theory will read:
"Energy can be approached from the aspect of frequency or the aspect of wave length. One is the inverse of the other. Each vibra tion of energy is divided into aliquot parts, through the smaller prime numbers. The aliquot parts di vides the categories of waves into classes which
16
Quantum Arithmetic
are harmonic, one with another, in a quantum way."
In order to understand energy, one must work with the individual wave in a train of waves. It can be done through frequency or through wave length. One is the inverse of the other. Each type of wave is unique. It gains its uniqueness through one prime number which is unique to that wave. Since there are generally up to seven prime num bers contributing to each wave, the other six will contribute to the aliquot part of that wave. That aliquot part is usually larger in value than the unique prime number. The aliquot part is equal to the product of these six. In the complete wave, it is repeated the number of times indicated by the unique prime number. (See Harmonic Cycle).
VARIOUS MYRIADS OF HARMONY
Harmony involves two waves which have the same aliquot part. When two or more waves have the same aliquot parts, these aliquot parts will act in the same way that two or more gears will mesh by having teeth at the same pitch. The diameter of each gear will represent the unique prime number of a wave. Such gears will depict a mechanical harmony.
In music, one's ears can be trained to discrim inate tone and harmony between tones. In color, one can also be trained to discriminate in harmo ny of color. But relying on the senses can often be misleading, partly because different people dis criminate in different ways, which is apparent in tone deafness, and in color blindness.
In chemistry, harmony appears to play a great part in chemical combinations. The harmony ap pears to be between the electrons in harmonic en ergy states. That is the reason that most chemical reactions require rather specific temperatures to place electrons in those specific energy bands which are harmonic.
Harmony appears to play a large part in mat ters of health. This harmony has more to do with maintaining the standing wave patterns which maintain the body in workable and harmonious condition. More will be written on this.
In astronomy, the harmony is well recognized between the planets in our Solar system. This ap pears to carry one myriad upward into harmony between stars in a galaxy. Then it carries upward still further, by a myriad, to a harmony and bal ance between galaxies.
In mathematics, the harmony is demonstrat
ed, in a different way, between the different geo metric shapes. But between differently propor tioned shapes of the same figure, direct harmony can be traced through each separate Koenig se ries, and each series of quantum numbers. The Fibonacci series is only one of the quantum num ber series.
Each of the myriads can contrib ute some knowledge of energy and harmony, but music and chemistry show the greatest promise.
THEORY OF HARMONY
The theory of harmony is: "Harmo nies, or harmony occurs between two dissimilar cy cles of energy, when both can be divided into simi lar aliquot parts having the same magnitude but different multitudes.”
Each of the dissimilar cycles is identified by its one unique prime factor which is not a factor within the aliquot part. Each unique prime is used as a multiplier of the aliquot part, in relation to the other cycle. The two cycles will produce a new wave equal to the product of their two unique prime factors.
In ordinary music, this longer wave will ordinarily be above the threshold of hearing, in the myriad below audible sound, (re ferred to later as the "mental myriad"). This ap pears to be one of the reasons that music can have an emotional content. It also plays a promi nent part in many of the ancient legends involving music, such as the harp of Hermes; the pipes of Pan; the conch shells of Aruna, in the Mahabhar ata; and even the trumpets which crumbled the walls of Jericho. These are all examples of harmo ny and discord.
THEORY OF ENERGY FORMS
Energy, itself, still is not defined. However, some of the mechanics of energy can be theorized, after the energy is created by the "forc es". The theory is as follows: "Newly formed ener gy is created in cycles, called Iota, assumed to be at about four quadrillion hertz. All cycles are equal but will aggregate in various magnitudes of “Har monic Cycles”. These shorter magnitudes of Har monic Cycles will aggregate in groups of three to seven to form, (cascade), into a longer wave to be measured in a larger units, in continuing progres sion.
This opposes the contention of Lord Rayleigh that waves propagate to increasing frequency. The smaller waves appear to aggregate
17
Quantum Arithmetic
in groups of five to seven to produce longer waves
which are between 5,000 to 10,000 times the length of the aggregating waves. These longer waves are then measured in the original lower prime numbers.
How or where the cascading occurs is largely undetermined. The cause seems to be “Quantum Flexibility” which is discussed elsewhere.
Any given longer wavelength is composed of the original shorter wave length which becomes an aliquot part. This aliquot part is multiplied by a prime number greater than 7, to create the long er wave. A major part of any aliquot part is its par ticular harmonic cycle. The parts of the larger wave are this harmonic cycle within the aliquot part, and its unique prime number multiplier.
Energy filters downward instead of upward, and this is the essence of entropy. When the ali quot parts of two different cycles are similar to the extent that they are in low fractional ratio in their multitudes, they may cascade downward several octaves or even one or more myriads. It is thought there are seven useable octaves in a myriad, but there may be as many as twelve octaves. Instabili ty occurs in the higher octaves.
NOTE: "myriad" and "iota" will be described in later chapters.
EXPERIMENTATION
The preceding remarks, and theories are based on experiments which will be described. These ex periments are based on judgemental physical re sults, and more-so, on non-judgemental use of the mathematical features of Quantum Arithmetic as described heretofore. A few of these experiments will be described for others to replicate.
The judgemental part, which are based upon sensory impressions in each case, which may dif fer from person to person. The non-judgemental part of the results derives from application of the proven principles of Quantum Arithmetic.
HARMONY
Most experiments involved the use of the Com modore-64 computer to produce specific musical frequencies and waveforms. The choice of this computer is made because it has the SID, (Sound Interface Device), chip which permits producing frequencies or wavelengths digitally, without re sorting to the analog slicing of a sine wave. Wave lengths were carried as near as possible, to a mil lionth of one hertz, (9 significant digits).
Several experiments were per formed to determine the numerical content of notes which were in good harmony.
One such experiment entered a single note in a music program of my own deriva tion. This note was produced at a constant tone. Then a second note was introduced at a shorter wavelength, which was run through a (one mil lionth hertz) incremental loop to increase its wave length in very small increments until it became 50% longer wavelength than the original constant note.
These 2-note chords progressed through “difference” beat-tones, until they died out at harmony. Passing through harmony, they began the “summation” beat tones and then dis harmony. There were certain values of the incre mental note which produced excellent harmonic response, with the constant tone. The program was modified to show the decimal ratio between the two notes on the video screen. The best har monic response occurred at the low fractional ra tios between the two notes.
Unexpected accidents occurred in that the sound, at a reasonable volume, produced harmonics which destroyed the electronics in the video monitor. After the third attempt, and the third video monitor I had the speaker disconnect ed and used a remote speaker.
Because harmonics seemed to dic tate the lower prime numbers, (2, 3, 5 & 7), I de signed the stable tone to a product of those low prime numbers but did not revise the variable tone. Then knowing the prime factors of the steady tone, I added a subroutine to the program, which would factor any variable tone at the point of my selection, and send them to the printer. Se lecting both harmonic and discord pairs of notes, I found the discord notes were often very high prime numbers, and often only a single prime number. Those variable tones which produced the best harmony, invariably factored into a 2, a 3, a 5 and/or a 7, or their powers, and one to four higher prime numbers. The lower the largest prime number in the factors, the more harmoni ous was the 2-note chord judged to be. I thus con cluded that the lowest prime numbers produced the best harmonic chords.
CATTLE PROBLEM
Secondly, realizing that the num bers up to 7 seemed to indicate the best harmony, it occurred to me that Archimedes had used these
18
Quantum Arithmetic
fractions up to 7ths, in his famous Cattle Prob lem. I determined to derive some sort of solution to this problem based on the numbers of bulls, which were derived by Wurm in the last century. The numbers of the cows solved out to be decimal values up to ten thousandths, of a COW. I ration alized that possibly some cows were not all one pure color.
Then having these as a solution for eight enti ties, I entered them in the computer as a musical scale. They were all played, each note with each of the other seven notes as 2-note chords. To my surprise they were either curiously harmonic or were curiously, slightly inharmonic, as in a minor chord. This Cattle Problem then, may have been the relative tones of the “Music of the Spheres”, or possibly, a variation of some ancient wave theory.
I-CHING
In other reading on mathematics from ancient China, I came across the Book of Permutations. This book also had the eight characters, of which four were male and four were female. One of the males, Chien, was the leading male character, as the yellow bulls are a leading character in the problem of Archimedes. Both cases have 4 males and 4 females. They were in a different setting but the values in I-Ching were given as individual bi nary hexagrams. In combinations of 64 pairs of them they were claimed to produce certain results for purposes of prognostication.
A musical score could not be produced with only eight notes, so I took the eighteen low frac tional values, up to 7ths, of each of these eight notes. At a certain pitch value these 144 notes had a selection of notes which would produce a scale which paralleled the chromatic scale within one or two hertz for each note. Using this scale to produce several popular songs on the computer, I found that there were some very unusual harmon ics in these songs.
The music was pleasant, but listeners would invariably complain, after listening for several minutes. I found that, for several hours, after lis tening to these unusual chords, it would be im
possible for me to relax and sleep.
NOT ARCHIMEDES?
With these unusual results, I con cluded this Cattle Problem must be a variation of the Music of the Spheres given to us by Archi medes. The results were very much in line with ancient legends of Sirens, Harp of Hermes, Pipes of Pan, and hundreds of other references. But with the I-Ching originating several thousand years before Archimedes, indicates that this prob lem was derived by Archimedes from translations of manuscripts which were then in the Library of Alexandria. Eratosthenes, who was their librarian possibly had had the translation made and ad vised Archimedes. Then Archimedes on setting the translation in the Cattle Problem, addressed the problem to Eratosthenes. Possibly neither one of them would have known of the harmonic content of the problem.
QUANTUM PITCH
There is one major unknown in this music and that is knowledge of the proper pitch. This music must be played at a very precise pitch in order to replicate the frequencies which apply to the quantumness of nature. It is true that the video was destroyed three times. But of the millions of 2-note chords which were played, it required only one specific chord to destroy the electronics in the video. This music will not de stroy any walls of Jericho. But one must realize that this musical experimentation can create dan ger.
CASCADES
Another musical area was also in vestigated. This concerned the production of low tones from ultrasound. The computer would only produce frequencies up to four kilohertz. So one note at about 3.9 kilohertz was entered, and an other, (harmonious), lower note at about 3 kilo hertz was entered. Both of these were steady notes. Then a third variable note was played with them. In the incremental loop, its value varied from the second note to the first. Playing through
19
MUSIC OF THE SPHERES
Quantum Arithmetic
PART II
taves. This scale has real potential, but has not been fully investigated, nor understood. The read
This brings us to “Music Of The Spheres” which, also, originated long before Pythagoras. “Music of the Spheres”, or “Song Celestial” is the subject of Archimedes problem “The Cattle Of Thrinacia”. In Odyssey of Ulysses they are called “Cattle Of The Sun”. Quantum Arithmetic dictates that the final solution consists of eight integers, all below the value of 10,000.
The values of the "male" notes are already inte gers. The remaining "female" notes will be within 0.002 per unit of an integer. Given those parame ters, a solution is impossible to derive with con ventional arithmetic. It is not impossible with Quantum Arithmetic, but it will be difficult until the full knowledge of this system of mathematics is better understood.
The four “male” notes are 891, 1580, 1602 and 2226. Their factors are: 2, 3, 5 & 7, along with one larger prime number between 7 and 100. This is in line with all of Quantum Arithmetic, and in deed, with chemistry, and astronomy which shall be demonstrated soon. Factors of these four notes are: 891= 34, 11; 1602= 2, 32, 89; 1580= 22, 5, 79; 2226 =2, 3, 7, 53.
The female notes used were 754.95383, 1050.7297, 1197.965 and 1547.4254. Simplifying them to 756, 1050, 1197 & 1548 gives us 756= 22x33x7; 1050=2x3x52x7; 1197=32x7x19; 1548=22x32x43. The primary basis of harmony lies in the numbers 2, 3, 5 and 7, and three or fewer larger primes, as factors. These integer values for the female notes all factor into the primes 2, 3, 5, 7, and one larger prime. The discrepancy between the integer plus decimal, and the factored integer is within the “Flexibility Factor” which is described later. That is the basis for their inclusion in Music of the Spheres.
A MUSICAL SCALE?
These are only eight keynotes. Fractions of halves to sevenths of these notes must be added to each keynote to form a musical scale of 18 tones for that keynote. For the eight keynotes there will be a full scale of 144 tones. Since the keynotes harmonize, most of the secondary notes of each scale will also harmonize. Generated in this way, the 144 notes will have values from 796 to about 4000 falling within two and one half oc
er can take it from there.
HISTORY
Music has always been an art rather than a science. Under Quantum Arithmetic it becomes possible to study music as a mathematical sci ence. Pythagoras introduced music as a science 2500 years ago in his working with strings of dif ferent lengths, relating them to the prime right tri angles. According to Evans G. Valens, in “The Number of Things” pg. 24, Pythagoras, while at the “Temple of The Muses” in Crotona, Italy, believed that, “an orderly universe based upon numerical ratios and numbers, could account for the harmo nious reality which underlies the confused ap pearance of the universe”. Today, our science is torn between the chaos of the Universe and the extreme order of it. The latter applies particularly to things at the molecular scale. Quantum Arith metic is finding that order applies throughout.
That this order existed everywhere, seems to have been discovered many millennia before Py thagoras. Since harmony derives from the frac tional relationships, from halves to sevenths, as used by Archimedes I looked at the relationships more closely. The Cattle Problem which he pro posed, contained these same fractions, but it went further than that. It contained three stages of fractions, of fractions, of fractions, in a sort of cat aract. With that in mind I proceeded to work with the Cattle Problem from a musical point of view. I calculated all fourths, fifths, sixths and sevenths of each note and added this to the notes.
Using the values for the 18 note scales of each bull, which I had derived, I set these up as wave lengths and synthesized the tones. In pairs, they produced good harmony. Then deriving scales for the cows, these were added to the first four, mak ing 144 tones. These male and female tones to gether, produced more harmonious and resonant chords.
AND I-CHING
The setting of I-Ching and the Cattle is differ ent but the characters are the same. The Cattle Problem gives complex proportions, one to an other. But the Chinese setting gives each charac ter a binary value. It then characterizes each
20
Quantum Arithmetic
character as having an influence in a specific area. It then places them in different pairs making a chord which is harmonious or discord in its in fluence. The two are different in these respects. One is subjective while the other is objective. There can be little doubt that these two settings evolve from the same natural source, and that source is Music of the Spheres. Their applications seem entirely different but both societies were aware of Music of the Spheres.
In China it was called “Song Celestial”. In Odyssey, Ulysses lands his ship on the shore of an island where the sacred “Cattle of the Sun” are grazing, which is several hundred years before Py thagoras. The Book of Permutations from China, precedes the Cattle Problem by more than 3,000 years. And from India, Mahabharata (about 4,000 B. C.), refers often to “Song Celestial”. This indi cates that the Cattle Problem was not a direct der ivation by Archimedes.
Eratosthenes or Archimedes derived it from manuscripts which were in the Library of Alexan dria. Archimedes derived the problem and sent it to Librarian, Eratosthenes at the Library of Alex andria. The Library was sacked and burned be tween 50 B. C. and 640 A.D. The paper was found 1500 years later in the basement of the Vatican. But it is quite possible that neither Eratosthenes nor Archimedes knew the importance of the trans lation which he, or Eratosthenes, had made.
In the context we come up with today, the Cat tle Problem seems, to be a simple statement of what we call wave theory, and theory of harmon ics. These seem to be, eight keynote frequencies of eight different musical scales. With this thought, I expanded each of these eight notes to 18 for each musical scale. That gave a total of 144 tones. Each, tone would be expanded nearly one octave by having it generate other tones which were greater than itself by 6/7 to 1/7; by 5/6 to 1/6; by 4/5 to 1/5; and by 3/4 and 1/4. The halves and thirds are taken care of in these higher fractions.
By synthesizing and playing these 144 tones in pairs, they generated many beautiful harmoni ous chords similar to the chords between the key notes.
BEATS
Within these chords were various tremolo rates which made them extraordinarily pleasant. There were very few, true discords. The discords were notorious, mainly for the faster reverbera tions, (or beats), which they created between cer tain pairs, creating unpleasant feelings. They also
caused the electronics used in their creation, to reverberate. That was when the video monitor blew up. These very low tone beats seemed to gen erate a sympathetic response within the electron ics and within one's nervous system, particularly when there appeared to be harmony.
When the beat tone was less than six beats per second they generated pleasant feelings. From six to about 12 beats per second they generated feelings of activity. From twelve to twenty beats per second there was definite unease, if not actual panic. This response came not from the auditory senses but seemed to come directly as sympathet ic bodily response. Other researchers have done work in this area and confirmed there is an emo tional or mental effect. Above twenty-five beats per second the beats, (degenerated?), turned into bass tones in a cascade explained at the begin ning of this chapter.
This range of vibratory waves, (or beats), from approximately 0.2 hertz to 30 hertz, becomes one of the Myriad scales. The Mental Myriad is formed as the energy cascades from the Musical Myriad of 10,000 different musical frequencies, to these lower, beat, frequencies. Edgar Cayce said, many times that music provides the opening to under standing all other parts of science. It appears that that opening is beginning to present itself.
COMPOSITIONS
Selecting from the 144 notes, one set of notes which closely approximated three octaves of the chromatic scale, I preceded to play familiar songs with this scale. The songs appeared to be quite or dinary, except for one thing. That thing was the reverberation between notes when they appeared as two-note or three-note chords. Casual listeners enjoyed these songs for periods of only a few min utes. Longer than that, they found them objec tionable, but could not define the cause of their objection. The variations in the reverberation, or beat, or some would call it the tremolo in the chords, was the cause. Listening to such a song for thirty minutes was sufficient to prevent sleep for several hours until the nerves had quieted down.
Choosing a different system of chords for a more uniform tremolo, removed that objection so long as the beats remained below four beats per second.
In this musical scale the formation of possible chords is tremendous. Any note can make an har monious chord with nearly any other note. There are literally millions of such chords in, say, a 52
21
Quantum Arithmetic
note scale. The problem lies in the unacceptable reverberation rate within certain of them.
PITCH
This still is not the Music of the Spheres. In order to be Music of the Spheres, it must be played at a very precise pitch for the basic scale. This, of course, is to be expected because quan tum vibrations must be very precise. This is par ticularly true when one observes the wavelengths of the spectrographic lines of the elements. The unperturbed waves are at a precise and unchang ing pitch.
Spacing between the different notes in Music of the Spheres, is also quite variable, very much like the spacing between the spectrographic lines of the elements. There are no “half tones”. In the span between two of our normal notes there may appear as many as four or five notes in this scale. Or there may be a gap in the scale in which there is no note in a span of four half-tones.
Every piano tuner is familiar with the problem that no piano can be tuned to perfect octaves and still have uniform spacing between the half-tones. To that extent, when a piano is tuned by thirds or by fifths, the piano tuner must employ a certain amount of judgement. In Music of the Spheres, it appears that the perfect octave is sacrificed in fa vor of maintaining the order of the notes, and the quantumness of the scale.
In any scale of Music of the Spheres, the key note cannot be distinguished. This brings up a question, “How do our ears know that a keynote is a keynote?” Are we able to relate certain ratios be tween notes of the scale, and the order in which those ratios fall? Then from this order of ratios, can we pick out the specific ratio which leads to the keynote? This seems to be a possibility. Does one learn to recognize a keynote, -- the note on which the song should end, - or is this an inborn capability? To an untrained ear, a keynote played by itself is unrecognizable from any other note. It is recognized only at the end of a series of notes, to provide a basis for recognition.
The capability of the complete scale of Music of the Spheres, appears to justify many of the an cient legends and stories of events in ancient times which involve music. There are hundreds of such examples: Greek Myth; Biblical; in the Hindu writings; and elsewhere. I-Ching implies that some of these specific vibratory rates are bombarding us to create the tendencies which it describes. One scientist, a member of International Keely Society, is investigating this possibility. He will eventually
tell of his findings.
POETRY AND DANCE
Music has its tones; its vibrato; its beat; its measure; its phrases; and its stanzas. Each one can be considered as an aliquot part of the one following.
Poetry has very much the same organization. Each letter of a poem is like a note of music. Each syllable, each word, each meter or phrase, and each stanza follow in progression. Each is an ali quot part of the next. There are different meters in which a poem can be written, and that meter must be maintained for a specific period. The me ter can be changed within a poem but a given me ter must be maintained in most cases. Changing the meter changes the whole context of a piece of poetry. In effect, poetry is "quantum", and must maintain that quantumness order to be recog nized as a valid poem.
Dance is the physical expression of music. A dancer on a stage can be likened to the playing of a flute solo. In a room full of dancers, with all dancing in pairs, (male and female), to a given dance pattern, the dancers maintain a uniform but dynamic pattern, in their steps and gyrations. The beauty and enjoyment of dancing is in its quantumness.
A room full of dancers can be likened to the movement of particles within an atom or a mole cule. They are dancing in unison, (proton and neutron), with the music, with the poetic lyrics of the music, and with the music itself, all of which is quantumness in its dynamics, and coordina tion. The musicians are playing, but let them skip one note, or one beat, and the whole scene, turns into chaos. The quantumness is broken.
That people fall so naturally into this recogni tion of tune, time, and meter, seems to indicate there is an inborn capacity to recognize quantum ness. This recognition can be enhanced by addi tional training. But to what extent is each person born with this capability?
It required nearly eight years to complete the sequences described above. The steps in the order they occurred:
(1) The research indicated that the primes 2, 3, 5 & 7 were important
(2) These numbers were used as fractions by Archimedes in the Cattle Problem, so it was solved.
22
Quantum Arithmetic
(3) These eight values were set up as relative tones. They proved harmonic.
(4) Each tone was expanded to 18 notes for a total of 144 notes.
(5) A chromatic scale was made.
WAVE NUMBERS
In harmonics, we should have learned that when two quantum numbers utilize four or more of the same prime factors, (within the four inte gers of the quantum number), they are HARMON IC. In addition to these four factors, they may have two or three more prime numbers in their makeup, one of which is unique to each wave. The prime numbers 2 and 3 must be in each, and they may also share a 5 or a 7 or both.
Perfect harmony is obtained only in nature. Man has been unable to create this perfect harmo ny, probably for the causes described below. Na ture's harmony is found in the planets and stars. It is also found in the atoms of every element, with each element having its own harmony. It is found in sound and in visible light. The specific harmony describes the item to which it is related.
How do we find the musical scale of this per fect harmony? We have already found the part of the harmony, which we call chords of music, by some natural sense of music. We can also see the harmony in such places as the rainbow. But until now, we have been unable to put true number val ues to this harmony.
Each wavelength of energy has its own specific quantum number (b, e, d, a). The quantum num ber represents a specific ellipse or set of ellipses which generate that wavelength. The wavelength of energy must be represented in natural units in stead of our invented units. Those natural num bers are derived through "quantizing" of the em pirical data.
The empirical measurements of an ellipse usu ally give the perigee and apogee, but an empirical ellipse can also be quantized with only the elliptic ity given. In this case the quantization is relative. When quantizing from this empirical data, the natural quantum measurements can be regarded as relative to each other. Ellipticity is a ratio which is an analog of the ratios, b/a and e/d. They can be returned to conventional units at any time. This relative quantum number depends upon the “truth factor” which is used. (See Line 10, "y2", of program “Quantize”).
TRUTH FACTOR
At the lowest truth value, the quantum num ber comes within reason, but it should be able to return to within the last two digits of the input data. It depends on the validity of the assumption of accuracy of the empirical data. Empirical data which is not accurate within 1% cannot be quan tized. (i.e. Moon orbit cannot be quantized be cause it is too inaccurate.)
The variation between the empirical and the quantum data should not change more than 0.01%. The integers of the quantum number should all be less than 100, with a correct truth factor. If too severe a truth factor is used the quantum number will be too high and will factor differently.
FACTORING
Say we take two integers, 128 and 64. They 2/128 & 1/64 will factor the same. Then add one unit in each case. This will become 2/129 and 1/65. These are basically the same number values in relation to all others, under a given truth factors, but they will factor entirely differently, 2/129 = 2/(3x43), but 1/65 = 1/(5x13). The 65 is much more dependable because it has lower prime numbers. That is critical in determining the aliquot parts.
ALIQUOT PARTS
Each wave can be broken down into aliquot units of: 6-units, (2x3); 30-units, (2x3x5); 42 units, (2x3x7); Or 210-units, (2x3x5x7). These ali quot partial waves are the most fundamental breakdown, and can be considered as Harmonic Cycles. The most significant breakdown will prob ably be some prime multiple of 30 or 42.
For further explanation one must refer to the “Rubber Band” hypothesis which is posed in Vol ume III of Pythagoras And The Quantum World, (1985). This hypothesis rates the specific areas of stable vs. weak quantum status over very short bandwidths. Each quantum wave can be pictured as riding in its own trough, or bandwidth. The lower the point in the trough, that is, the lower the prime factors, the stronger will be the quan tum. Between adjacent troughs is a peak, over which a quantum jump can be made. It is a case of stability of a given quantum frequency. Stabili ty, rather than strength, is the leading parameter. In practical application, any ellipse will be pulled or pushed out of this central position by perturba tion. The “Rubber Band Hypothesis” derives from
23
Quantum Arithmetic
a mathematical feature showing strength of quan tization and not strength of the energy. It will re quire further empirical testing to verify what sort of truth value is to be used, and how to obtain it. Quantization and empirical testing must go hand in-hand with each assisting the other.
There are two variables in Program Quantize which affect the truth of calculated relativity be tween two items.
It is for this reason that the correct truth value must be used. That truth value should give re sults somewhere between 1:5000 and 1:10000.
When a spectrogram is taken of any chemical, or element, the lines are usually given in ang stroms. This is the wavelength of the "color" of each spectrographic line. They are used here as they are represented in “CRC Handbook of Chem istry and Physics” (1973). These can be assumed to represent the ellipticity of the ellipse which that generates color wavelength. The ellipse is taken to be the elliptical path which an electron is follow ing, at its respective energy state. In the case de scribed below the ellipse must be considered to be only a “picture” of the relative mathematical rela tionship between colors.
Any energy state is relative to all other energy states, so long as the input data remains uniform. The separation between spectral lines will then represent the quantum changes in frequencies of energies between states of an electron.
In a single atom, all energy states are not oc cupied simultaneously, and an electron does not necessarily jump to the state which adjoins it in the table. Likewise, the difference in quantum numbers represents a quantum change in energy frequencies. They become relative in natural quantum units, just as the empirical measure ments were relative, (in angstroms), but were not quantum.
This is accomplished through Program Quan tize. The input will be the ellipticity in decimal form for the perigee and unity for the apogee.
THE QUANTUM ELLIPSE
A short review of the previous chapter in Book #1, will show that for any ellipse, the semi-major diameter must be a square number. If the semi major diameter is not the square of an, integer, the ellipse can be expanded or shrunk until it be comes a square integer, (D). (That is to say, the unit of measure is changed to some abstract, nat ural unit of measure.) After the correct square
number for the semi-major diameter is found, then the products of its square root, (d), and three other integers, (b, e, & a), will give all other meas urements, including the other three measure ments along the major diameter. These will be: db, de, d2 and da, retaining them in Fibonacci configuration. The four integers b, e, d, & a, dic tate the natural quantum number for that quan tum ellipse. More detail is given in Books #1 & #2.
The exact same procedure can be followed without quantizing, but then none of the meas urements can be factored and prime numbers cannot be recognized. (See Problems 7 & 8 on page 15 of Book 2.)
In the natural Quantum Number, e & d must be coprime; and since Euclid Book VII, Proposi tion 28, claims that their sum (a), and their differ ence (b), will be prime to both e & d this quantum number will be unique and differentiated from all other spectral lines which have their own unique Quantum Number.
BREAKING DOWN THE WAVE
After the natural Quantum Number is derived, the four integers of that quantum number are fac tored to further break them down into their prime factors. Every Quantum Number will have the fac tors 2, and 3 represented in it. It should also have either a 5 or a 7, or both, among its factors. It may have other prime factors up to a total of sev en, for males, and up to six for females. Male quantum numbers may have a minimum of four factors. Females may have a minimum of three prime factors. These factors are the origin of the quantumness in nature.
TRUTH DETERMINATION
All spectrographic lines of twenty of the most common elements were Quantized. After the Quantum Numbers were derived, and factored, the factors of sodium and chlorine were cata logued in a data set. The quantum numbers for Hydrogen, Carbon & Oxygen, (hydrocarbons), were in another set.
Sodium and Chlorine, using a truth value of one part per million had five correlating prime numbers in each, (2x3x5x13x47). At a truth of one part per million, Chlorine Line #38 quantized, with factors of 2, 3, 5, 7, 13, 47 and missed the input check by 0.001 angstrom. Sodium line #15 quantized at 94, 265, 359, 624, (which was far too high). It had the factors 2, 3, 5, 13, 47, 53, 359. The Chlorine and Sodium both had the common factors of 2, 3, 5, 13 & 47, with Chlorine having 7
24
Quantum Arithmetic
as its unique factor. Sodium had 53 as its unique factor. An extra factor 359, appeared as a factor. The 359 should have been 360 whose factors in crease the powers of the 2, 3 & 5 and eliminates the 359. This analysis of waves is entirely mathe matical.
The assumed truth was too high. Again, Sodi um and Chlorine were quantized at a reduced truth factor of one part in 40,000. The quantiza tion is right but the empirical data still has more error than allowed. It appears that the truth factor should be one part in 5040. Y should be 1/71 which equals 1/√5040. These factors were put through the data set to find correlating lines. The quantization and the correlation of factors are shown at the end of this chapter.
The truth level is critical.
An original quantizing was done at an accura cy of one part per million. At that level, other pa rameters enter the picture to cloud the outcome.
The tables at the end of this chapter are based on a truth value of 1 part in 40,000. All spectro graphic lines are included to demonstrate what can be done.
EXAMPLES
Spectral lines of a hydrogen energy state corre lated with spectral lines of an oxygen energy state in five different cases. The eight energy lines of hy drogen showed three cases of utmost harmony within themselves. (Not shown).
One can hypothesize that correlation between a certain energy state of one element, and a cer tain energy state of another may form a bonding between the two of them at this point. The bond ing should occur when there is a correlation be tween aliquot parts of the two energy states. Quantization can locate these energy states, but it must be done correctly and then substantiated, if possible by empirical testing.
QUANTIZED STATES
The last pages of this chapter consist of quan tization and classification of all energy states of sodium and chlorine. These are made available to demonstrate what can be done with Quantum Arithmetic. Certain assumptions can be made, and some of the weak points can be explored.
The first four pages are the quantization of So dium electron states, and the factors of those states. The first two pages are the quantization,
and factoring. The 3rd and 4th pages give the fac tors, first in the order of increasing wavelengths, then in the order of their factors.
The next six pages are the same order for Chlorine. The last two pages combine Sodium and Chlorine prime factors in the order of increasing size of the prime factors. Somewhere in these last two pages should give some indication of the bonding point of Sodium and Chlorine to form ta ble salt, NaCl.
The quantization is accomplished by using an assumed unit of measure which is 10,000 ang stroms. In this unit, all spectrographic lines be come decimals for the perigee of an ellipse, and one unit is used for the apogee. This is the input, which is given in angstrom units. After that is giv en the angstrom units are reconstructed from the quantization. This is given to indicate that no troubles have occurred in the calculation. It con sists of N/Q x 10,000.
The largest wavelength is slightly less than 2,000 angstroms. The 10,000 used above would probably give more reliable results if the unit of measure had been taken as 4,000 angstroms maximum, so we would be working between the fractions of zero to 1/2. Harmonics and Music of The Spheres indicate these would have been a better choice. As it is the quantizations occur be tween zero and 1/5 for N/Q. The relativity between energy states still exists, but the prime factors have come out differently than would have oc curred with the lower unit.
EVALUATION
On the last page of the tables, Sodium is giv en as quantizing Line 301.32 angstroms, with the prime factors of 2, 3, 5, 7, 19, 23 & 83. Chlorine quantizes Line 1565.05 angstroms with the prime factors 2, 3, 5, 7, 19, 23 & 97.
They differ by only the very last prime factor. From previous indications, these two lines will harmonize very strongly. They will have sympa thetic vibration between them according to these calculations.
There are many other such harmonics. In the table: (Page 41 & 42) line 23 & 24; line 30 & 31; line 41 & 42; line 45 & 46; line 49 & 50; line 53 & 54; 58 & 59; and many others will harmonize in pairs.
One can suppose this correlation between these two energy states of electrons may form a bonding, between sodium and chlorine, at this
25
Quantum Arithmetic
point. Whether this conclusion is correct or not, this example demonstrates what can be done to furnish more mathematically precise information within chemistry. This effectively, takes out the empiricism, and eliminates many uncertainties, if proper judgement is used.
CONCLUSION
This demonstration of breaking empirical waves into their aliquot parts, was made in con nection with chemistry. The same process will ap ply, without change, in waves of any denomina tion be they color or heat or astronomic cycles or musical tones.
An original ambient wave, as found in nature, is never a true sine wave. Many of them can ap proach the shape of a sine wave, but will be mod ulated by smaller waves. The complete wave will closely resemble the path of a planet as it is per turbed from its orbit by gravitational pull from nu merous other bodies. The other bodies will, like wise, be pulled or pushed from their perfect orbits by the first.
A sinusoidal projection of Earth's elliptical or bit, will show waverings, where the Earth is pushed out of its ideal orbit. This occurs about 13 times each year by the Moon which shoves the earth aside by 3,000 miles. Other smaller devia tions are caused by the planets. The annual orbit of Earth will project into a sine wave, but the sine curve will have irregularities. The same thing will apply to Mars and Venus our nearest neighbors. Each of these will have waverings similar to that of Earth. In matching that of Earth they will have the same applicable prime numbers. These mutu al waverings will act as cogs on a wheel, like gears in a clock, tying the whole planetary system to gether. Each gear will have a prime number of “teeth”.
When Quantum Numbers are assigned to the orbits of all planets, their minor modulations will match with others nearby. This is the basis of har mony. The minor modulation cycles of an individ ual orbit or wave represents its unique “harmon ic”. When two or more such waves or cycles have enough similar prime number values, they can be said to be “in harmony”.
TABLES
The following pages show the quantization of 42 spectrographic lines for Sodium, and 80 lines for Chlorine. The lines are given, in both cases, in the order of their frequency.
The first line shows: The element; the se quence number of the energy state; The quantum number of that line; The empirical value for that line, in angstroms; And the derived quantum val ue. The second line gives the factors of the quan tum number, for determination of its aliquot parts.
The next step on pages 37 & 38, correlates the factors and lists them in the order they were quantized. Page 38 then reorders them in a data set. Pages 39 through 45 does the same for the 80 energy states of Chlorine. Pages 46, 47 & 48 cor relates the energy states of Sodium and Chlorine by their factors. Those lines which have the most identical sets of factors in the two data sets, will appear together. These sets of factors are still not the true set because the truth factor is still set at one magnitude too high.
These tables should be considered only as a demonstration of what can be done with Quan tum Arithmetic. The prime factors for many lines are still above 100 and in the area of instability. In this demonstration, the truth value is still too strict, and the value of the unit of measure, (10,000 angstroms per unit), certainly is not cor rect.
The empirical measurement in Angstroms is clearly better than metric measurement. But if the length of one angstrom is not within one-tenth percent of what the natural measure should be it will lead to errors in quantization.
At this point Quantum Arithmetic with its ab solutism is far ahead of empirical research. Em pirical research can be upgraded with the applica tion of Quantum Arithmetic and that is the next step in making progress.
There are many features which become appar ent in these tables. Deficiencies in knowledge of the true values of natural parameters prevents at tainment of the required absolutes in Quantum Arithmetic. Deficiencies in instrument calibration for empirical research will often lead to wrong di agnoses. But the two, working together can achieve phenomenal progress.
In the present state, quantization can indicate where harmony probably exists between widely different frequencies. But the natural state re quires more accuracy on all fronts, whether it is in the range of visible light, the ranges of audible and ultrasound, and even in the ranges of astron omy. It may even show us the way in the ranges of magnetism and gravity.
26
Quantum Arithmetic
At the more strict truth value, (one part per million), the empirical wavelength matches pre cisely with the quantum wavelength, but the quantum number is far too high to be reasonable. When the truth value of empirical values is re duced to one part in 10,000, the match between the empirical and natural values will vary but the quantum numbers still will not factor within the requirements of Quantum Arithmetic. It becomes apparent that the unit to be used in these tables is one part in 5040 units.
The reader may note that when a factor is car ried to a power only the base prime is entered in the data set. This is necessary in order to classify according to prime numbers only.
This same process is used with music to de rive the correct pitch for notes of Music of the
Spheres. In astronomy, the empirical values have been used. Here have been found flagrant errors in measurement. All of the planets quantized quite well but their satellites were too far in error to achieve quantization. This applies particularly to our own Moon. On reviewing past encyclopedia editions, McGraw-Hill, various editions vary wide ly in the distance to our own Moon. This occurs because two elliptical movements are involved. One movement is around the Earth, and the other is parallel to Earth axis.
27
Quantum Arithmetic
QUANTIZATION & factoring of 44 Spectrographic, Sodium Lines
SODIUM # I QUANT #= 6 97 103 200 ANGSTROM= 300.15 (TRUE= 300) FACTORS= 16 3 25 97 103 ( 1 )
SODIUM # 2 QUANT #= 6 97 103 200 ANGSTROM= 300.2 (TRUE= 300) FACTORS= 16 3 25 97 103 ( 1 )
SODIUM # 3 QUANT #= 10 161 171 332 ANGSTROM= 301.32 (TRUE= 301.204819) FACTORS= 8 9 5 7 19 23 83 ( 1 )
SODIUM # 4 QUANT #= 11 177 188 365 ANGSTROM= 301.44 (TRUE= 301.369863) FACTORS= 4 3 5 11 47 59 73 ( 1 )
SODIUM # 5 QUANT #= 1 16 17 33 ANGSTROM= 302.45 (TRUE= 303.030303) FACTORS= 16 3 11 17 ( 1 )
SODIUM # 6 QUANT #= 1 13 14 27 ANGSTROM= 372.08 (TRUE= 370.37037) FACTORS= 2 27 7 13 ( I )
SODIUM # 7 QUANT #= 9 115 124 239 ANGSTROM= 376.38 (TRUE= 376.569038) FACTORS= 4 9 5 23 31 149 ( 1 )
SODIUM # 8 QUANT #= 11 37 48 85 ANGSTROM= 1293.97 (TRUE= 1294.11765) FACTORS= 16 3 5 11 17 37 ( 1 )
SODIUM # 9 QUANT #= 15 49 64 113 ANGSTROM= 1327.74 (TRUE= 1327.43363) FACTORS= 64 3 5 49 113 ( I )
SODIUM # 10 QUANT #= 19 61 80 141 ANGSTROM= 1347.54 (TRUE= 1347.51773) FACTORS= 16 3 5 19 47 61 ( 1 )
SODIUM # 11 QUANT #= 22 69 91 160 ANGSTROM= 1374.89 (TRUE= 1375) FACTORS= 64 3 5 7 11 13 23 ( 1 )
SODIUM # 12 QUANT #= 17 52 69 121 ANGSTROM= 1404.68 (TRUE= 1404.95868) FACTORS= 4 3 121 13 17 23 ( 1 )
SODIUM # 13 QUANT #= 32 91 123 214 ANGSTROM= 1495.21 (TRUE= 1495.3271) FACTORS= 64 3 7 13 41 107 ( 1 )
SODIUM # 14 QUANT #= 19 54 73 127 ANGSTROM= 1496.01 (TRUE= 1496.06299) FACTORS= 2 27 19 73 127 ( 1 )
SODIUM # 15 QUANT #= 31 88 119 207 ANGSTROM= 1497.73 (TRUE= 1497.58454) FACTORS= 8 9 7 11 17 23 31 ( 1 )
SODIUM # 16 QUANT #= 11 31 42 73 ANGSTROM= 1506.41 (TRUE= 1506.84932) FACTORS= 2 3 7 11 31 73 ( 1 )
SODIUM # 17 QUANT #= 11 31 42 73 ANGSTROM= 1506.91 (TRUE= 1506.84932) FACTORS= 2 3 7 11 31 73 ( 1 )
SODIUM # 18 QUANT #= 41 115 156 271 ANGSTROM= 1513.1 (TRUE= 1512.91513 FACTORS= 4 3 5 13 23 41 149 ( 1 )
SODIUM # 19 QUANT #= 19 53 72 125 ANGSTROM= 1519.63 (TRUE= 1520) FACTORS= 8 9 125 19 53 ( 1 )
SODIUM # 20 QUANT #= 31 78 109 187 ANGSTROM= 1657.92 (TRUE= 1657.75401) FACTORS= 2 3 11 13 17 31 109 ( 1 )
SODIUM # 21 QUANT #= 35 81 116 197 ANGSTROM= 1776.57 (TRUE= 1776.64975) FACTORS= 4 81 5 7 29 149 ( 1 )
SODIUM # 22 QUANT #= 16 37 53 90 ANGSTROM= 1778.24 (TRUE= 1777.77778) FACTORS= 32 9 5 37 53 ( 1 )
SODIUM # 23 QUANT #= 23 53 76 129 ANGSTROM= 1783.04 (TRUE= 1782.94574) FACTORS= 4 3 19 23 43 53 ( 1 )
28
Quantum Arithmetic
SODIUM # 24 QUANT #= 37 85 122 207 ANGSTROM= 1787.19 (TRUE= 1787.43961) FACTORS= 2 9 5 17 23 37 61 ( 1 )
SODIUM # 25 QUANT #= 44 101 145 246 ANGSTROM= 1788.85 (TRUE= 1788.61789) FACTORS= 8 3 5 11 29 41 101 ( 1 )
SODIUM # 26 QUANT #= 25 57 82 139 ANGSTROM= 1798.41 (TRUE= 1798.56115) FACTORS= 2 3 25 19 41 139 ( 1 )
SODIUM # 27 QUANT #= 29 66 95 161 ANGSTROM= 1801.26 (TRUE= 1801.24224 ) FACTORS= 2 3 5 7 11 19 23 29 (1)
SODIUM # 28 QUANT #= 15 34 49 83 ANGSTROM= 1807.09 (TRUE= 1807.22892) FACTORS= 2 3 5 49 17 83 ( 1 )
SODIUM # 29 QUANT #= 34 77 111 188 ANGSTROM= 1808.38 (TRUE= 1808.51064) FACTORS= 8 3 7 11 17 37 47 ( 1 )
SODIUM # 30 QUANT #= 45 101 146 247 ANGSTROM= 1821.7 (TRUE= 1821.86235) FACTORS= 2 9 5 13 19 73 101 ( 1 )
SODIUM # 31 QUANT #= 53 118 171 289 ANGSTROM= 1833.87 (TRUE= 1833.91003) FACTORS= 2 9 17 17 19 53 59 ( 1 )
SODIUM # 32 QUANT #= 49 109 158 267 ANGSTROM= 1835.22 (TRUE= 1835.20599) FACTORS= 2 3 49 89 109 149 ( 1 )
SODIUM # 33 QUANT #= 9 20 29 49 ANGSTROM= 1837.89 (TRUE= 1836.73469) FACTORS= 4 9 5 49 29 ( 1 )
SODIUM # 34 QUANT #= 14 31 45 76 ANGSTROM= 1841.82 (TRUE= 1842.10526) FACTORS= 8 9 5 7 19 31 ( 1 )
SODIUM # 35 QUANT #= 19 42 61 103 ANGSTROM= 1845.02 (TRUE- 1844.66019) FACTORS= 2 3 7 19 61 103 ( 1 )
SODIUM # 36 QUANT #= 5 11 16 27 ANGSTROM= 1850.15 (TRUE= 1851.85185) FACTORS= 16 27 5 11 ( 1 )
SODIUM # 37 QUANT #= 5 11 16 27 ANGSTROM= 1851.19 (TRUE= 1851.85185) FACTORS= 16 27 5 11 ( 1 )
SODIUM # 38 QUANT #= 5 11 16 27 ANGSTROM= 1853.17 (TRUE= 1851.85185) FACTORS= 16 27 5 11 ( 1 )
SODIUM # 39 QUANT #= 28 61 89 150 ANGSTROM= 1866.45 (TRUE= 1866.66667) FACTORS= 8 3 25 7 61 89 ( 1 )
SODIUM # 40 QUANT #= 6 13 19 32 ANGSTROM= 1873.37 (TRUE= 1875) FACTORS= 64 3 13 19 ( 1 )
SODIUM # 41 QUANT #= 6 13 19 32 ANGSTROM= 1875.08 (TRUE= 1875) FACTORS= 64 3 13 19 ( 1 )
SODIUM # 42 QUANT #= 51 110 161 271 ANGSTROM= 1881.91 (TRUE= 1881-.91882) FACTORS= 2 3 5 7 11 17 23 149 (1 )
SODIUM # 43 QUANT #= 46 99 145 244 ANGSTROM= 1885.09 (TRUE= 1885.2459) FACTORS= 8 9 5 11 23 29 61 ( 1 )
SODIUM # 44 QUANT #= 33 71 104 175 ANGSTROM= 1885.74 (TRUE= 1885.71429) FACTORS= 8 3 25 7 11 13 71 ( 1 )
29
Quantum Arithmetic
DATA SET OF FACTORS OF 44 SODIUM LINES -- (By wavelength) Element Line Angstrom Factor 3 5 7 11 13 17 19 23
Sodium #01 0300.15 2 3 5 97 103
Sodium #02 0300.20 2 3 5 97 103
Sodium #03 0301.32 2 3 5 7 19 23 83
Sodium #04 0301.44 2 3 5 11 47 59 73
Sodium #05 0302.45 2 3 11 17
Sodium #06 0372.08 2 3 7 13
Sodium #07 3746.38 2 3 5 23 31 6.3.13 Faulty factors Sodium #08 1293.97 2 3 5 11 17 37
Sodium #09 1327.74 2 3 5 7 113
Sodium #10 1347.54 2 3 5 19 47 61
Sodium #11 1374.69 2 3 5 7 11 13 23
Sodium #12 1404.68 2 3 11 13 17 23
Sodium #13 1495.21 2 3 7 13 41 107
Sodium #14 1496.01 2 3 19 73 127
Sodium #15 1497.73 2 3 7 11 17 23 31
Sodium #16 1506.41 2 3 7 11 31 73
Sodium #17 1506.91 2 3 7 11 31 73
Sodium #18 1543.10 2 3 5 13 23 41 6.9.5 Faulty factors Sodium #19 1519.63 2 3 5 19 53
Sodium #20 1657.92 2 3 11 13 17 31 109
Sodium #21 1776.57 2 3 5 7 29 197
Sodium #23 1783.04 2 3 19 23 43 53
Sodium #22 1778.24 2 3 5 37 53
Sodium #24 1787.19 2 3 5 17 23 37 61
Sodium #25 1788.85 2 3 5 11 29 41 101
Sodium #26 1798.41 2 3 5 19 41 139
Sodium #27 1801.26 2 3 5 7 11 19 23 29 Sodium #28 1807.09 2 3 5 7 17 83
Sodium #29 1808.38 2 3 7 11 17 37 47
Sodium #30 1821.70 2 3 5 13 19 73 101
Sodium #31 1833.87 2 3 17 19 53 59
Sodium #32 1835.22 2 3 7 79 89 109
Sodium #33 1837.89 2 3 5 7 29
Sodium #34 1841.82 2 3 5 7 19 31
Sodium #35 1845.02 2 3 7 19 61 103
Sodium #36 1850.15 2 3 5 11
Sodium #37 1851.19 2 3 5 11
Sodium #38 1853.17 2 3 5 11
Sodium #39 1866.45 2 3 5 7 61 89
Sodium #40 1873.37 2 3 13 19
Sodium #41 1875.08 2 3 13 19
Sodium #42 1881.91 2 3 5 7 11 17 23 271 Sodium #43 1885.09 2 3 5 11 23 29 61
Sodium #44 1885.74 2 3 5 7 11 13 71
30
Quantum Arithmetic
DATA SET OF 44 SODIUM LINES (by Precedence of Factors)
Element Line Angstrom Factor--) 3 5 7 11 13 17 19 23
Sodium #12 1404.68 2 3 11 13 17 23
Sodium #20 1657.92 2 3 11 13 17 31 109
Sodium #05 0302.45 2 3 11 17
Sodium #40 1873.37 2 3 13 19
Sodium #41 1875.08 2 3 13 19
Sodium #31 1833.87 2 3 17 19 53 59
Sodium #23 1783.04 2 3 19 23 43 53
Sodium #14 1496.01 2 3 19 73 127
Sodium #15 1497.73 2 3 7 11 17 23 31
Sodium #29 1808.38 2 3 7 11 17 37 47
Sodium #17 1506.91 2 3 7 11 31 73
Sodium #16 1506.41 2 3 7 11 31 73
Sodium #06 0372.08 2 3 7 13
Sodium #13 1495.21 2 3 7 13 41 107
Sodium #35 1845.02 2 3 7 19 61 103
Sodium #32 1835.22 2 3 7 79 89 109
Sodium #38 1853.17 2 3 5 11
Sodium #36 1850.15 2 3 5 11
Sodium #37 1851.19 2 3 5 11
Sodium #08 1293.97 2 3 5 11 17 37
Sodium #43 1885.09 2 3 5 11 23 29 61
Sodium #25 1788.85 2 3 5 11 29 41 101
Sodium #04 0301.44 2 3 5 11 47 59 73
Sodium #30 1821.70 2 3 5 13 19 73 101
Sodium #18 1513.10 2 3 5 13 23 41 6.9.5 Faulty Sodium #24 1787.19 2 3 5 17 23 37 61
Sodium #26 1798.41 2 3 5 19 41 139
Sodium #10 1347.54 2 3 5 19 47 61
Sodium #19 1519.63 2 3 5 19 53
Sodium #07 0376.38 2 3 5 23 31 6.3.13 Faulty Sodium #22 1778.24 2 3 5 37 53
Sodium #02 0300.20 2 3 5 97 103
Sodium #01 0300.15 2 3 5 97 103
Sodium #11 1374.69 2 3 5 7 11 13 23
Sodium #44 1885.74 2 3 5 7 11 13 71
Sodium #42 1881.91 2 3 5 7 11 17 23 271 Sodium #27 1801.26 2 3 5 7 11 19 23 29 Sodium #09 1327.74 2 3 5 7 113
Sodium #28 1807.09 2 3 5 7 17 83
Sodium #03 0301.32 2 3 5 7 19 23 83
Sodium #34 1841.82 2 3 5 7 19 31
Sodium #33 1837.89 2 3 5 7 29
Sodium #21 1776.57 2 3 5 7 29 197
Sodium #39 1866.45 2 3 5 7 61 89
Corrected in later run
31
Quantum Arithmetic
QUANTIZATION & Factoring of 80 Spectrographic Chlorine Lines
CHLORINE # 1 QUANT #= 9 76 85 161 ANGSTROM= 559.305 TRUE= 559.006211 FACTORS= 4 9 5 7 17 19 23 ( 1 )
CHLORINE # 2 QUANT #= 4 33 37 70 ANGSTROM= 571.904 TRUE= 571.428572 FACTORS= 8 3 5 7 11 37 ( 1 )
CHLORINE # 3 QUANT #= 5 41 46 87 ANGSTROM= 574.406 TRUE= 574.712644 FACTORS= 2 3 5 23 29 41 ( 1 )
CHLORINE # 4 QUANT #= 1 8 9 17 ANGSTROM= 586.24 TRUE= 588.235294 FACTORS= 8 9 17 ( 1 )
CHLORINE # 5 QUANT #= 17 129 146 275 ANGSTROM= 618.057 TRUE= 618.181818 FACTORS= 2 3 25 11 17 43 73 ( 1 )
CHLORINE # 6 QUANT #= 16 121 137 258 ANGSTROM= 619.982 TRUE= 620.155039 FACTORS= 32 3 121 43 137 ( 1 )
CHLORINE # 7 QUANT #= 16 121 137 258 ANGSTROM= 620.298 TRUE= 620.155039 FACTORS= 32 3 121 43 137 ( 1 )
CHLORINE # 8 QUANT #= 21 157 178 335 ANGSTROM= 626.735 TRUE= 626.865672 FACTORS= 2 3 5 7 67 89 139 ( 1 )
CHLORINE # 9 QUANT #= 11 81 92 173 ANGSTROM= 635.881 TRUE= 635.83815 FACTORS= 4 81 11 23 139 ( 1 )
CHLORINE # 10 QUANT #= 17 125 142 267 ANGSTROM= 636.626 TRUE= 636.70412 FACTORS= 2 3 125 17 71 89 ( 1 )
CHLORINE # 11 QUANT #= 11 79 90 169 ANGSTROM= 650.894 TRUE= 650.887574 FACTORS= 2 9 5 11 169 139 ( 0 )
CHLORINE # 12 QUANT #= 13 92 105 197 ANGSTROM= 659.811 TRUE= 659.898477 FACTORS= 4 3 5 7 13 23 139 ( 1 )
CHLORINE # 13 QUANT #= 18 127 145 272 ANGSTROM= 661.841 TRUE= 661.764706 FACTORS= 32 9 5 17 29 127 ( 1 )
CHLORINE # 14 QUANT #= 24 169 193 362 ANGSTROM= 663.074 TRUE= 662.983425 FACTORS= 16 3 169 139 139 ( 1 )
CHLORINE # 15 QUANT #= 6 41 47 88 ANGSTROM= 682.053 TRUE= 681.818182 FACTORS= 16 3 11 41 47 ( 1 )
CHLORINE # 16 QUANT #= 13 88 101 189 ANGSTROM= 687.656001 TRUE= 687.830688 FACTORS= 8 27 7 11 13 101 ( 1 )
CHLORINE # 17 QUANT #= 7 47 54 101 ANGSTROM= 693.594 TRUE= 693.069308 FACTORS= 2 27 7 47 101 ( 1 )
CHLORINE # 18 QUANT #= 5 32 37 69 ANGSTROM= 725.271001 TRUE= 724.637682 FACTORS= 32 3 5 23 37 ( 1 )
CHLORINE # 19 QUANT #= 14 89 103 192 ANGSTROM= 728.951001 TRUE= 729.16666-, FACTORS= 64 2 3 7 89 103 ( 1 )
CHLORINE # 20 QUANT #= 14 83 97 180 ANGSTROM= 777.562001 TRUE= 777.777778 FACTORS= 8 9 5 7 83 97 ( 1 )
CHLORINE # 21 QUANT #= 13 76 89 165 ANGSTROM= 787.58 TRUE= 787.878788 FACTORS= 4 3 5 11 13 19 89 ( 1 )
32
Quantum Arithmetic
CHLORINE # 22 QUANT #= 25 146 171 317 ANGSTROM= 788.74 TRUE= 788.643533 ) FACTORS= 2 9 25 19 73 139 2 ( 1 )
CHLORINE # 23 QUANT #= 5 29 34 63 ANGSTROM= 793.342001 TRUE= 793.650794 ) FACTORS= 2 9 5 7 17 29 ( 1 )
CHLORINE # 24 QUANT #= 11 60 71 131 ANGSTROM= 839.297 TRUE= 839.694657 FACTORS= 4 3 5 11 71 131 ( 1 )
CHLORINE # 25 QUANT #= 11 60 71 131 ANGSTROM= 839.598999 TRUE= 839.694657 FACTORS= 4 3 5 11 71 131 ( 1 )
CHLORINE # 26 QUANT #= 9 49 58 107 ANGSTROM= 841.41 TRUE= 841.121495 FACTORS= 2 9 49 29 107 ( 1 )
CHLORINE # 27 QUANT #= 27 145 172 317 ANGSTROM= 851.691 TRUE= 851.735016 FACTORS= 4 27 5 29 43 139 2 ( 1 )
CHLORINE # 28 QUANT #= 23 118 141 259 ANGSTROM= 888.026 TRUE= 888.030888 FACTORS= 2 3 7 23 37 47 59 ( 1 )
CHLORINE # 29 QUANT #= 21 107 128 235 ANGSTROM= 893.549 TRUE= 893.617021 FACTORS= 64 2 3 5 7 47 107 ( 1 )
CHLORINE # 30 QUANT #= 10 47 57 104 ANGSTROM= 961.499 TRUE= 961.538461 FACTORS= 16 3 5 13 19 47 ( 1 )
CHLORINE # 31 QUANT #= 29 135 164 299 ANGSTROM= 969.92 TRUE= 969.899666 FACTORS= 4 27 5 13 23 29 41 ( 1 )
CHLORINE # 32 QUANT #= 18 83 101 184 ANGSTROM= 978.284 TRUE= 978.26087 FACTORS= 16 9 23 83 101 ( 1 )
CHLORINE # 33 QUANT #= 2 9 11 20 ANGSTROM= 998.372 TRUE= 1000 FACTORS= 8 9 5 11 ( 1 )
CHLORINE # 34 QUANT #= 2 9 11 20 ANGSTROM= 998.432 TRUE= 1000 FACTORS= 8 9 5 11 ( 1 )
CHLORINE # 35 QUANT #= 2 9 11 20 ANGSTROM= 1002.346 TRUE= 1000 FACTORS= 8 9 5 11 ( 1 )
CHLORINE # 36 QUANT #= 30 133 163 296 ANGSTROM= 1013.664 TRUE= 1013.51351 FACTORS= 16 3 5 7 19 37 139 ( 1 )
CHLORINE # 37 QUANT #= 8 35 43 78 ANGSTROM= 1025.553 TRUE= 1025.64103 ) FACTORS= 16 3 5 7 13 43 ( 1 )
CHLORINE # 38 QUANT #= 5 21 26 47 ANGSTROM= 1063.831 TRUE= 1063.82979 ) FACTORS= 2 3 5 7 13 47 ( 1 )
CHLORINE # 39 QUANT #= 11 46 57 103 ANGSTROM= 1067.945 TRUE= 1067.96117 FACTORS= 2 3 11 19 23 103 ( 1 )
CHLORINE # 40 QUANT #= 6 25 31 56 ANGSTROM= 1071.036 TRUE= 1071.42857 ) FACTORS= 16 3 25 7 31 ( 1 )
CHLORINE # 41 QUANT #= 6 25 31 56 ANGSTROM= 1071.767 TRUE= 1071.42857 ) FACTORS= 16 3 25 7 31 ( 1 )
CHLORINE # 42 QUANT #= 20 83 103 186 ANGSTROM= 1075.23 TRUE= 1075.26882 FACTORS= 8 3 5 31 83 103 ( 1 )
33
Quantum Arithmetic
CHLORINE # 43 QUANT #= 15 62 77 139 ANGSTROM= 1079.08 TRUE= 1079.13669 FACTORS= 2 3 5 7 11 31 139 ( 1 )
CHLORINE # 44 QUANT #= 9 37 46 83 ANGSTROM= 1084.667 TRUE= 1084.33735 FACTORS= 2 9 23 37 83 ( 1 )
CHLORINE # 45 QUANT #= 28 115 143 258 ANGSTROM= 1085.171 TRUE= 1085.27132 FACTORS= 8 3 5 7 11 13 23 43 ( 1 )
CHLORINE # 46 QUANT #= 28 115 143 258 ANGSTROM= 1085.304 TRUE= 1085.27132 FACTORS= 8 3 5 7 11 13 23 43 ( 1 )
CHLORINE # 47 QUANT #= 21 86 107 193 ANGSTROM= 1088.06 TRUE= 1088.0829 FACTORS= 2 3 7 43 107 139 ( 1 )
CHLORINE # 48 QUANT #= 23 94 117 211 ANGSTROM= 1090.271 TRUE= 1090.04739 FACTORS= 2 9 13 23 47 139 ( 1 )
CHLORINE # 49 QUANT #= 12 49 61 110 ANGSTROM= 1090.982 TRUE= 1090.90909  FACTORS= 8 3 5 49 11 61 ( 1 )
CHLORINE # 50 QUANT #= 13 53 66 119 ANGSTROM= 1092.437 TRUE= 1092.43698 FACTORS= 2 3 7 11 13 17 53 ( 1 )
CHLORINE # 51 QUANT #= 15 61 76 137 ANGSTROM= 1094.769 TRUE= 1094.89051 FACTORS= 4 3 5 19 61 137 ( 1 )
CHLORINE # 52 QUANT #= 15 61 76 137 ANGSTROM= 1095.148 TRUE= 1094.89051 FACTORS= 4 3 5 19 61 137 ( 1 )
CHLORINE # 53 QUANT #= 16 65 81 146 ANGSTROM= 1095.662 TRUE= 1095.89041 FACTORS= 32 81 5 13 73 ( 1 )
CHLORINE # 54 QUANT #= 16 65 81 146 ANGSTROM= 1095.797 TRUE= 1095.89041 FACTORS= 32 81 5 13 73 ( 1 )
CHLORINE # 55 QUANT #= 17 69 86 155 ANGSTROM= 1096.81 TRUE= 1096.77419 FACTORS= 2 3 5 17 23 31 43 ( 1 )
CHLORINE # 56 QUANT #= 18 73 91 164 ANGSTROM= 1097.369 TRUE= 1097.56098 FACTORS= 8 9 7 13 41 73 ( 1 )
34
Quantum Arithmetic
QUANTIZATION & Factoring of 80 Spectrographic Chlorine Lines
CHLORINE # 57 QUANT #= 113 77 96 173 ANGSTROM= 1098.068 TRUE= 1098.2659 FACTORS= 32 3 7 11 19 139 ( 1 )
CHLORINE # 58 QUANT #= 21 85 106 191 ANGSTROM= 1099.523 TRUE= 1099.47644 FACTORS= 2 3 5 7 17 53 139 ( 1 )
CHLORINE # 59 QUANT #= 1 4 5 9 ANGSTROM= 1107.528 TRUE= 1111.11111 FACTORS= 4 9 5 ( 1 )
CHLORINE # 60 QUANT #= 9 35 44 79 ANGSTROM= 1139.214 TRUE= 1139.24051 FACTORS= 4 9 5 7 11 139 ( 0 ) Faulty
CHLORINE # 61 QUANT #= 37 140 177 317 ANGSTROM= 1167.148 TRUE= 1167.19243 FACTORS= 4 3 5 7 37 59 139 2 ( 1 ) Faulty
CHLORINE # 62 QUANT #= 23 86 109 195 ANGSTROM= 1179.293 TRUE= 1179.48718 FACTORS= 2 3 5 13 23 43 109 ( 1 )
CHLORINE # 63 QUANT #= 17 63 80 143 ANGSTROM= 1188.774 TRUE= 1188.81119 FACTORS= 16 9 5 7 11 13 17 ( 1 )
CHLORINE # 64 QUANT #= 3 11 14 25 ANGSTROM= 1201.353 TRUE= 1200 FACTORS= 2 3 25 7 11 ( 1 )
CHLORINE # 65 QUANT #= 37 120 157 277 ANGSTROM= 1335.726 TRUE= 1335.74007 FACTORS= 8 3 5 37 139 8 3 13 ( 1 )
CHLORINE # 66 QUANT #= 19 61 80 141 ANGSTROM= 1347.24 TRUE= 1347.51773 FACTORS= 16 3 5 19 47 61 ( 1 )
CHLORINE # 67 QUANT #= 5 16 21 37 ANGSTROM= 1351.657 TRUE= 1351.35135 FACTORS= 16 3 5 7 37 ( 1 )
CHLORINE # 68 QUANT #= 6 19 25 44 ANGSTROM= 1363.447 TRUE= 1363.63636 FACTORS= 8 3 25 11 19 ( 1 )
CHLORINE # 69 QUANT #= 7 22 29 51 ANGSTROM= 1373.116 TRUE= 1372.54902 FACTORS= 2 3 7 11 17 29 ( 1 )
CHLORINE # 70 QUANT #= 8 25 33 58 ANGSTROM= 1379.528 TRUE= 1379.31034 FACTORS= 16 3 25 11 29 ( 1 )
CHLORINE # 71 QUANT #= 41 127 168 295 ANGSTROM= 1389.693 TRUE= 1389.83051 FACTORS= 8 3 5 7 41 59 127 ( 1 )
CHLORINE # 72 QUANT #= 31 96 127 223 ANGSTROM= 1389.957 TRUE= 1390.13453 FACTORS= 32 3 31 127 139 223 ( 1 ) Faulty
CHLORINE # 73 QUANT #= 25 77 102 179 ANGSTROM= 1396.527 TRUE= 1396.64804 FACTORS= 2 3 25 7 11 17 139 ( 1 ) Faulty
CHLORINE # 74 QUANT #= 32 95 127 222 ANGSTROM= 1441.47 TRUE= 1441.44144 FACTORS= 64 3 5 19 37 127 ( 1 )
CHLORINE # 75 QUANT #= 35 97 132 229 ANGSTROM= 1528.569 TRUE= 1528.38428 FACTORS= 4 3 5 7 11 97 139 ( 1 ) 229 Faulty
CHLORINE # 76 QUANT #= 27 74 101 175 ANGSTROM= 1542.942 TRUE= 1542.85714 FACTORS= 2 27 25 7 37 101 ( 1 )
CHLORINE # 77 QUANT #= 24 65 89 154 ANGSTROM= 1558.144 TRUE= 1558.44156 FACTORS= 16 3 5 7 11 13 89 ( 1 )
35
Quantum Arithmetic
CHLORINE # 78 QUANT #= 36 97 133 230 ANGSTROM= 1565.05 TRUE= 1565.21739 FACTORS= 8 9 5 7 19 23 97 ( 1 )
CHLORINE # 79 QUANT #= 26 57 83 140 ANGSTROM= 1857.488 TRUE= 1857.14286 FACTORS= 8 3 5 7 13 19 83 ( 1 )
CHLORINE # 80 QUANT #= 1 2 3 5 ANGSTROM= 1997.37 TRUE= 2000 FACTORS= 2 3 5 ( 1 )
Corrected later
36
Quantum Arithmetic
DATA SET OF FACTORS OF 80 CHLORINE LINES (By wavelength) (page 1 of 2) Element Line Angstrom Factors f3 f5 f7 f11+
chlorine #01 0559.305 2 3 5 7 17 19 23 chlorine #02 0571.904 2 3 5 7 11 37 chlorine #03 0574.406 2 3 5 23 29 41 chlorine #04 0586.240 2 3 17
chlorine #05 0618.057 2 3 5 11 17 43 73 chlorine #06 0619.982 2 3 11 43 137 chlorine #07 0620.298 2 3 11 43 137 chlorine #08 0626.735 2 3 5 7 67 89 139 chlorine #09 0635.881 2 3 11 23 139 chlorine #10 0636.626 2 3 5 17 71 89 chlorine #11 0650.894 2 3 5 11 13 139 chlorine #12 0659.811 2 3 5 7 13 23 139 chlorine #13 0661.841 2 3 5 17 29 127 chlorine #14 0663.074 2 3 13 41 139 chlorine #15 0682.053 2 3 11 41 47 chlorine #16 0687.656 2 3 7 11 13 101 chlorine #17 0693.594 2 3 7 47 101 chlorine #18 0725.271 2 3 5 23 37 chlorine #19 0728.951 2 3 7 89 103 chlorine #20 0777.562 2 3 5 7 83 97 chlorine #21 0787.580 2 3 5 11 13 19 89 chlorine #22 0788.740 2 3 5 19 73 139 chlorine #23 0793.342 2 3 5 7 17 29 chlorine #24 0839.297 2 3 5 11 71 131 chlorine #25 0839.599 2 3 5 11 71 131 chlorine #26 0841.410 2 3 7 29 107 chlorine #27 0851.691 2 3 5 29 43 139 chlorine #28 0888.026 2 3 7 23 37 47 59 chlorine #29 0893.549 2 3 5 7 47 107 chlorine #30 0961.499 2 3 5 13 19 47 chlorine #31 0969.920 2 3 5 13 23 29 41 chlorine #32 0978.284 2 3 23 83 101 chlorine #33 0998.372 2 3 5 11
chlorine #34 0998.432 2 3 5 11
chlorine #35 1002.346 2 3 5 11
chlorine #36 1013.664 2 3 5 7 19 37 139 chlorine #37 1025.553 2 3 5 7 13 43 chlorine #38 1063.831 2 3 5 7 13 47 chlorine #39 1067.945 2 3 11 19 23 103 chlorine #40 1071.036 2 3 5 7 31 chlorine #41 1071.767 2 3 5 7 31 chlorine #42 1075.230 2 3 5 31 83 103 chlorine #43 1079.080 2 3 5 7 11 31 139 chlorine #44 1084.667 2 3 23 37 83 chlorine #45 1085.171 2 3 5 7 11 13 23 43 chlorine #46 1085.304 2 3 7 11 13 23 43 chlorine #47 1088.060 2 3 7 43 107 139 chlorine #48 1090.271 2 3 13 23 47 139 chlorine #49 1090.982 2 3 5 7 11 61
37
Quantum Arithmetic
DATA SET OF FACTORS OF 80 LINES OF Chlorine (By Wavelength) Page 2 of 2 Element Line Angstrom Factors f3 f5 f7 f11+
chlorine #50 1092.437 2 3 7 11 13 17 53 chlorine #51 1094.769 2 3 5 19 61 137
chlorine #52 1095.148 2 3 5 19 61 137
chlorine #53 1095.662 2 3 5 13 73
chlorine #54 1095.797 2 3 5 13 73
chlorine #55 1096.810 2 3 5 17 23 31 43 chlorine #56 1097.369 2 3 7 13 41 73
chlorine #57 1098.068 2 3 7 11 19 139
chlorine #58 1099.523 2 3 5 7 17 53 139 chlorine #59 1107.528 2 3 5
chlorine #60 1139.214 2 3 5 7 11 79
chlorine #61 1167.148 2 3 5 7 37 59 317 chlorine #62 1179.293 2 3 5 13 23 43 109 chlorine #63 1188.774 2 3 5 7 11 13 17 chlorine #64 1201.353 2 3 5 7 11
chlorine #65 1335.726 2 3 5 37 157 277
chlorine #66 1347.240 2 3 5 19 47 61
chlorine #67 1351.657 2 3 5 7 37
chlorine #68 1363.447 2 3 5 11 19
chlorine #69 1373.116 2 3 7 11 17 29
chlorine #70 1379.528 2 3 5 11 29
chlorine #71 1389.693 2 3 5 7 41 59 127 chlorine #72 1389.957 2 3 31 127 223
chlorine #73 1396.527 2 3 5 7 11 17 179 chlorine #74 1441.470 2 3 5 19 37 127
chlorine #75 1528.569 2 3 5 7 11 97 229 chlorine #76 1542.942 2 3 5 7 37 101
chlorine #77 1558.144 2 3 5 7 11 13 89 chlorine #78 1565.050 2 3 5 7 19 23 97 chlorine #79 1857.488 2 3 5 7 13 19 83 chlorine #80 1997.370 2 3 5
38
Quantum Arithmetic
DATA SET OF FACTORS OF 80 CHLORINE LINES (By Precedence of Factors/1 of 2) Element Line Angstrom Factors f3 f5 f7 f1l+
I chlorine #39 1067.945 2 3 11 19 23 103 2 chlorine #09 0635.881 2 3 11 23 139
3 chlorine #15 0682.053 2 3 11 41 47
4 chlorine #07 0620.298 2 3 11 43 137
5 chlorine #06 0619.982 2 3 11 43 137
6 chlorine #48 1090.271 2 3 13 23 47 139 7 chlorine #14 0663.074 2 3 13 41 139
8 chlorine #04 0586.240 2 3 17
9 chlorine #44 1084.667 2 3 23 37 83
10 chlorine #32 0978.284 2 3 23 83 101
11 chlorine #72 1389.957 2 3 31 127 223
12 chlorine #50 1092.437 2 3 7 11 13 17 53 13 chlorine #46 1085.304 2 3 7 11 13 23 43 14 chlorine #16 0687.656 2 3 7 11 13 101 15 chlorine #69 1373.116 2 3 7 11 17 29 16 chlorine #57 1098.068 2 3 7 11 19 139 17 chlorine #56 1097.369 2 3 7 13 41 73 18 chlorine #28 0888.026 2 3 7 23 37 47 59 19 chlorine #26 0841.410 2 3 7 29 107
20 chlorine #47 1088.060 2 3 7 43 107 139 21 chlorine #17 0693.594 2 3 7 47 101
22 chlorine #19 0728.951 2 3 7 89 103
23 chlorine #80 1997.370 2 3 5
24 chlorine #59 1107.528 2 3 5
25 chlorine #33 0998.372 2 3 5 11
26 chlorine #35 1002.346 2 3 5 11
27 chlorine #34 0998.432 2 3 5 11
28 chlorine #11 0650.894 2 3 5 11 13 139 29 chlorine #21 0787.580 2 3 5 11 13 19 89 30 chlorine #05 0618.057 2 3 5 11 17 43 73 31 chlorine #68 1363.447 2 3 5 11 19
32 chlorine #70 1379.528 2 3 5 11 29
33 chlorine #25 0839.599 2 3 5 11 71 131 34 chlorine #24 0839.297 2 3 5 11 71 131 35 chlorine #30 0961.499 2 3 5 13 19 47 36 chlorine #31 0969.920 2 3 5 13 23 29 41
37 chlorine #62 1179.293 2 3 5 13 23 43 109 38 chlorine #53 1095.662 2 3 5 13 73
39 chlorine #54 1095.797 2 3 5 13 73
40 chlorine #55 1096.810 2 3 5 17 23 31 43 41 chlorine #13 0661.841 2 3 5 17 29 127 42 chlorine #10 0636.626 2 3 5 17 71 89 43 chlorine #74 1441.470 2 3 5 19 37 127 44 chlorine #66 1347.240 2 3 5 19 47 61 45 chlorine #51 1094.769 2 3 5 19 61 137 46 chlorine #52 1095.148 2 3 5 19 61 137 47 chlorine #22 0788.740 2 3 5 19 73 139 48 chlorine #03 0574.406 2 3 5 23 29 41 49 chlorine #18 0725.271 2 3 5 23 37
39
Quantum Arithmetic
DATA SET OF FACTORS OF 80 LINES OF CHLORINE (By Precedence of FACTORS/ 2 of 2 Element Line Angstrom Factors f3 f5 f7 fll+
50 chlorine #27 0851.691 2 3 5 29 43 139
51 chlorine #42 1075.230 2 3 5 31 83 103
52 chlorine #65 1335.726 2 3 5 37 157 277
53 chlorine #64 1201.353 2 3 5 7 11
54 chlorine #45 1085.171 2 3 5 7 11 13 23 43 55 chlorine #77 1558.144 2 3 5 7 11 13 89 56 chlorine #63 1188.774 2 3 5 7 11 13 17 57 chlorine #73 1396.527 2 3 5 7 11 17 179 58 chlorine #43 1079.080 2 3 5 7 11 31 139 59 chlorine #02 0571.904 2 3 5 7 11 37
60 chlorine #49 1090.982 2 3 5 7 11 61
61 chlorine #60 1139.214 2 3 5 7 11 79
62 chlorine #75 1528.569 2 3 5 7 11 97 229 63 chlorine #79 1857.488 2 3 5 7 13 19 83 64 chlorine #12 0659.811 2 3 5 7 13 23 139 65 chlorine #37 1025.553 2 3 5 7 13 43
66 chlorine #38 1063.831 2 3 5 7 13 47
67 chlorine #01 0559.305 2 3 5 7 17 19 23 68 chlorine #23 0793.342 2 3 5 7 17 29
69 chlorine #58 1099.523 2 3 5 7 17 53 139 70 chlorine #78 1565.050 2 3 5 7 19 23 97 71 chlorine #36 1013.664 2 3 5 7 19 37 139 72 chlorine #40 1071.036 2 3 5 7 31
73 chlorine #41 1071.767 2 3 5 7 31
74 chlorine #67 1351.657 2 3 5 7 37
75 chlorine #76 1542.942 2 3 5 7 37 101
76 chlorine #61 1167.148 2 3 5 7 37 59 317 77 chlorine #71 1389.693 2 3 5 7 41 59 127 78 chlorine #29 0893.549 2 3 5 7 47 107
79 chlorine #08 0626.735 2 3 5 7 67 89 139 80 chlorine #20 0777.562 2 3 5 7 83 97
40
Quantum Arithmetic
COMBINED DATA SETS OF SODIUM AND CHLORINE (Page 1 of 3) Element Line Angstrom Factors f3 f5 f7 f1l
1 Sodium #12 1404.680 2 3 11 13 17 23 2 Sodium #20 1657.920 2 3 11 13 17 31 109 3 Sodium #05 0302.450 2 3 11 17
4 chlorine #39 1067.945 2 3 11 19 23 103 5 chlorine #09 0635.881 2 3 11 23 139 6 chlorine #15 0682.053 2 3 11 41 47 7 chlorine #07 0620.298 2 3 11 43 137 8 chlorine #06 0619.982 2 3 11 43 137 9 Sodium #41 1875.080 2 3 13 19
10 Sodium #40 1873.370 2 3 13 19
11 chlorine #48 1090.271 2 3 13 23 47 139 12 chlorine #14 0663.074 2 3 13 41 139 13 chlorine #04 0586.240 2 3 17
14 Sodium #31 1833.870 2 3 17 19 53 59 15 Sodium #23 1783.040 2 3 19 23 43 53 16 Sodium #14 1496.010 2 3 19 73 127 17 chlorine #44 1084.667 2 3 23 37 83 18 chlorine #32 0978.284 2 3 23 83 101 19 chlorine #72 1389.957 2 3 31 127 223 20 chlorine #16 0687.656 2 3 7 11 13 101 21 chlorine #50 1092.437 2 3 7 11 13 17 53 22 chlorine #46 1085.304 2 3 7 11 13 23 43 23 Sodium #15 1497.730 2 3 7 11 17 23 31 24 chlorine #69 1373.116 2 3 7 11 17 29 25 Sodium #29 1808.380 2 3 7 11 17 37 47 26 chlorine #57 1098.068 2 3 7 11 19 139 27 Sodium #16 1506.410 2 3 7 11 31 73 28 Sodium #17 1506.910 2 3 7 11 31 73 29 Sodium #06 0372.080 2 3 7 13
30 Sodium #13 1495.210 2 3 7 13 41 107 31 chlorine #56 1097.369 2 3 7 13 41 73 32 Sodium #35 1845.020 2 3 7 19 61 103 33 chlorine #28 0888.026 2 3 7 23 37 47 59
34 chlorine #26 0841.410 2 3 7 29 107 35 chlorine #47 1088.060 2 3 7 43 107 139 36 chlorine #17 0693.594 2 3 7 47 101 37 Sodium #32 1835.220 2 3 7 79 89 109 38 chlorine #19 0728.951 2 3 7 89 103 39 chlorine #59 1107.528 2 3 5
40 chlorine #80 1997.370 2 3 5
41 Sodium #38 1853.170 2 3 5 11
42 chlorine #34 0998.432 2 3 5 11
43 Sodium #37 1851.190 2 3 5 11
44 chlorine #35 1002.346 2 3 5 11
45 chlorine #33 0998.372 2 3 5 11
46 Sodium #36 1850.150 2 3 5 11
47 chlorine #11 0650.894 2 3 5 11 13 139 48 chlorine #21 0787.580 2 3 5 11 13 19 89 49 Sodium #08 1293.970 2 3 5 11 17 37
41
Quantum Arithmetic
50 chlorine #05 0618.057 2 3 5 11 17 43 73 51 chlorine #68 1363.447 2 3 5 11 19 52 Sodium #43 1885.09 2 3 5 11 23 29 61 53 chlorine #70 1379.528 2 3 5 11 29 54 Sodium #25 1788.85 2 3 5 11 29 41 101
Note; Factors corrected
42
Quantum Arithmetic
COMBINED DATA SETS OF SODIUM AND CHLORINE (Page 2 of 3)
Element Line Angstrom Factors f3 f5 f7 f11+ 55 Sodium #04 0301.440 2 3 5 11 47 59 73 56 chlorine #24 0839.297 2 3 5 11 71 131 57 chlorine #25 0839.599 2 3 5 11 71 131 58 chlorine #30 0961.499 2 3 5 13 19 47 59 Sodium #30 1821.700 2 3 5 13 19 73 101 60 chlorine #31 0969.920 2 3 5 13 23 29 41 61 Sodium #18 1513.100 2 3 5 13 23 41 6.9.5 62 chlorine #62 1179.293 2 3 5 13 23 43 109 63 chlorine #54 1095.797 2 3 5 13 73 64 chlorine #53 1095.662 2 3 5 13 73 65 chlorine #55 1096.810 2 3 5 17 23 31 43 66 Sodium #24 1787.190 2 3 5 17 23 37 61 67 chlorine #13 0661.841 2 3 5 17 29 127 68 chlorine #10 0636.626 2 3 5 17 71 89 69 chlorine #74 1441.470 2 3 5 19 37 127 70 Sodium #26 1798.410 2 3 5 19 41 139 71 Sodium #10 1347.540 2 3 5 19 47 61 72 chlorine #66 1347.240 2 3 5 19 47 61 73 Sodium #19 1519.630 2 3 5 19 53 74 chlorine #52 1095.148 2 3 5 19 61 137 75 chlorine #51 1094.769 2 3 5 19 61 137 76 chlorine #22 0788.740 2 3 5 19 73 139 77 chlorine #03 0574.406 2 3 5 23 29 41 78 Sodium #07 0376.380 2 3 5 23 31 6.3.13 79 chlorine #18 0725.271 2 3 5 23 37 80 chlorine #27 0851.691 2 3 5 29 43 139 81 chlorine #42 1075.230 2 3 5 31 83 103 82 chlorine #65 1335.726 2 3 5 37 157 277 83 Sodium #22 1778.240 2 3 5 37 53 84 Sodium #01 0300.150 2 3 5 97 103 85 Sodium #02 0300.200 2 3 5 97 103 86 chlorine #64 1201.353 2 3 5 7 11 87 chlorine #63 1188.774 2 3 5 7 11 13 17 88 Sodium #11 1374.690 2 3 5 7 11 13 23 89 chlorine #45 1085.171 2 3 5 7 11 13 23 43 90 Sodium #44 1865.740 2 3 5 7 11 13 71 91 chlorine #77 1558.144 2 3 5 7 11 13 89 92 chlorine #73 1396.527 2 3 5 7 11 17 179 93 Sodium #42 1881.910 2 3 5 7 11 17 23 271 94 Sodium #27 1801.260 2 3 5 7 11 19 23 29 95 chlorine #43 1079.080 2 3 5 7 11 31 139 96 chlorine #02 0571.904 2 3 5 7 11 37 97 chlorine #49 1090.982 2 3 5 7 11 61 98 chlorine #60 1139.214 2 3 5 7 11 79 99 chlorine #75 1528.569 2 3 5 7 11 97 229 100 Sodium #09 1327.74 2 3 5 7 113 101 chlorine #79 1857.488 2 3 5 7 13 19 83 102 chlorine #12 0659.811 2 3 5 7 13 23 139 103 chlorine #37 1025.553 2 3 5 7 13 43 104 chlorine #38 1063.831 2 3 5 7 13 47
43
Quantum Arithmetic
105 chlorine #01 0559.305 2 3 5 7 17 19 23 106 chlorine #23 0793.342 2 3 5 7 17 29 107 chlorine #58 1099.523 2 3 5 7 17 53 139 108 Sodium #28 1807.09 2 3 5 7 17 83
44
Quantum Arithmetic
COMBINED DATA SETS OF SODIUM AND CHLORINE (page 3 of 3) Element Line Angstrom Factors f3 f5 f7 f11+
109 Sodium #03 0301.32 2 3 5 7 19 23 83 110 chlorine #78 1565.050 2 3 5 7 19 23 97 111 Sodium #34 1841.82 2 3 5 7 19 31 112 chlorine #36 1013.664 2 3 5 7 19 37 139 113 Sodium #33 1837.89 2 3 5 7 29 114 Sodium #21 1776.57 2 3 5 7 29 197 115 chlorine #40 1071.036 2 3 5 7 31 116 chlorine #41 1071.767 2 3 5 7 31 117 chlorine #67 1351.657 2 3 5 7 37 118 chlorine #76 1542.942 2 3 5 7 37 101 119 chlorine #61 1167.148 2 3 5 7 37 59 317 120 chlorine #71 1389.693 2 3 5 7 41 59 127 121 chlorine #29 0893.549 2 3 5 7 47 107 122 Sodium #39 1866.45 2 3 5 7 61 89 123 chlorine #08 0626.735 2 3 5 7 67 89 139 124 chlorine #20 0777.562 2 3 5 7 83 97
45
IOTA
Quantum Arithmetic
It seems that energy is thusly, created contin
ually and passes throughout the universe. If we
The term "Iota" has been used in previous pages. "Iota" is a Greek word meaning next to nothingness. It is a very smallest quantity.
Iota are defined herein as the basic unit of en ergy. The first Iota acts as a foundation and a core of energy, on which additonal Iota build them selves, or are directed to build themselves into dis crete groups.
The first, or core, Iota gathers around it other Iota in groups of 2, 3, 5, and 7 Iota, making what become Harmonic cycles. It becomes larger if the “2" is in fours instead of “twos”, or it can be in 8's, 16's and 9's and 27's, etc.
We can only theorize what an Iota may be. It is theorized to be a single precursor vibration, or a vibration of “one”. This rate of vibration is theor ized to be vibrating at a rate of four quadrillion vi brations per second. The period of such vibration is one 4-quadrillionth of a second. As they gather together in groups, the vibration rate becomes slower in accordance with the number of Iota in volved.
The basic unit Iota may be considered as a spark. Surrounding two Iota is one group of 3 Iota travelling in a circular path. These planetary groups are the “sparks” for making a larger group, an aliquot part. They become a nucleus.
Surrounding this nucleus may be another group of five, Iota travelling in pentagonal forma tion on a circular path. To this can be added, with or without the pentagonal array, another circuit of seven Iota, again on a circular path.
The picture so painted, comes directly from a mathematical context in the very beginning of Quantum Arithmetic. The mathematical context runs, 42 + 32 = 25. After negating the one core unit this becomes 24.
Following this, 52 - 12 = 24 and 72 - 52 = 24 making the magic number 24 which occurs throughout Quantum Arithmetic.
The “24" is divided into 2 parts, 3 parts and 4 parts, being 12, 8, and 6 respectively. Taking the outer rings of the Iota, 7 + 5 = 12. The two inner rings are 4 x 3 = 12, and 2 x 3 = 6. This begins the Koenig Series. After this original formation of Iota, every larger quantum energy group must contain a 2, a 3 and may have a 5 and/or a 7, and will be divisible evenly by this first one of these first Iota.
knew how to draw this energy, we could draw it freely and without limit.
Now that may sound quite impossible, but is it? So far as our scientific status is concerned we are familiar only with energy which is stored with in matter, and obtained through laws of matter. We store water in reservoirs to make electricity. We burn, wood, coal or petroleum products for energy. We use solar energy through specially composed elements into solar panels. But we have never found out where that energy comes from in the first place. It comes, mainly, from the Sun, but we know neither how it got there or how it gets here or in what form it travels. We only know it does travel and that, after it gets here, it be comes heat and light. Occasionally it will arrive as electrical static which we say is from Sun Spots. We know not of any case in which the energy ar rives in non-discrete chunks. It is all quantum in its nature, and it is quantum on the basis as de scribed above.
It was theorized that the unit vibaration was at a rate of 4 x 1015 vibrations per second. We must assume this second to be a human invented unit of measure until it can be determined other wise. This is not the ultimate period of vibration, per se.
It is the ultimate precursor energy vibration, which forms itself in waterfalls to aggregate in larger and larger chunks until we can recognize them. The precursor to this precursor energy unit is in the four forces, whatever they are. These forces accumulate and aggregate from 10-35 sec ond vibrations, much the same mathematical pro ceedure as the energy forms in larger chunks. But this field of forces is far beyond our capability, at least until we have firmed up and confirmed this proposed theory on energy. The speed at which the forces travel is considered to be instantane ous, but it would not be. It may travel a distance of a light year in a second but in all probability it is not instantaneous.
We have considered that energy travels at the speed of light which is approximately 186,000 miles per second. This probably is not true of en ergy. The shorter wavelengths of energy most probably travel much faster, and the much longer wavelengths, below infra-red wavelengths, prob ably travel much slower. We do know that blue light travels slightly faster than red light. This variation in speeds of travel probably carries throughout the energy spectrum. In the spectrum of the forces, above energy, the rate of travel of
46
Quantum Arithmetic
the forces is correspondingly much faster. DETAIL
The first Iota has a special facility in the math ematical aspects of formation of workable Iota. It corresponds to the hole in the center of the disks of gold in the Cattle problem, page 35, Book 1. It does not contribute to the magnitude of energy, just as the hole in the disk does not contribute to the amount of gold in each disk.
This is the answer to the question which was so often brought up in the volumes of “Pythagoras And The Quantum World”. The question was, “What happens to that one missing unit at the center?” That one unit which seems to be missing is transcendental. Its whole purpose is to set in motion the rules, both mathematical and physical, for creations, for which it becomes the foundation. In engineering design, that one unit is rather in consequential, but when one enters into quantum design, with its absolutism, this omission must be taken into consideration. It must be taken into consideration because a simple one unit change in a number, will completely change the quantization process, and the subsequent factoring.
Many Iota combine to form aliquot parts, but an aliquot part may not be greater than 10,000 Iota. This brings us to the Myriad.
THE MYRIAD
The Myriad is equal to 10,000. It was taken by the Greek philosophers as the highest number that need be considered in any calculation. In Quantum Arithmetic this is taken to be the maxi mum that any calculations need be considered. In effect, that is also the limit, (5 significant figures), that most design calculations need to be carried, even in conventional mathematics.
The Greek view was much more stringent. They considered that 7 was the highest prime number that need be considered. Seven Factorial (7!) is 5040, or the product of the first seven inte gers, lx2x3x4x5x6x7 = 5040. But Pythagoras con sidered going to 10 because 7x8x9xl0 is also 5040. So, when the Greeks reached 5040 they, also “rounded off” their calculations. But they had their reasons, as will be explained below.
NOTE: Archimedes demonstrated that the Greeks knew that numbers went on forever in his explanation of the “Sand Reconner” to the Ruler. It also demonstrated that these numbers were meaningless for our practical purposes.
For the waves, which we are considering, the length of a wave is determined by the size and number of prime numbers involved. That length is the product of all the prime numbers included in its quantum number. One can see the complexity which arises when it involves seven prime num bers. It becomes quite limiting when one realizes that 100 x 100 = 10,000. Nature has provide a so lution to this enigma.
THE LIMIT
Two vibrations coming together will produce a much longer vibration, as their product. Suddenly we find ourselves above the 10,000 limit of the Myriad.
How is it possible to work with waves of any length without going beyond this 10,000 limit?
There are two things to consider. The first was discussed in the use of only the lower prime num bers below 100 and preferably only prime num bers well below 60. This keeps the wavelength down.
The second is a realization of the instability or “error-factor” which ordinarily occurs in our cal culations. It becomes necessary to limit the num ber of decimal places to which we can trust our calculations. We may go to great ends, to extend to six or seven significant figures. This same thing happens in the quantum world, but in a much more predictable way.
QUANTUM FLEXIBILITY
What has been called "quantum", is not so absolute as the reader has been lead to believe. There is a flexibility in each unit. That flexibility has permitted mankind to perform non-quantum mathematics, and nonquantum design. But it is in using this flexibility zone which has brought contemporary science its troubles, chaos, and ac cidents. It is precisely this flexibility which per mits and promotes evolution and change in na ture.
The zone of flexibility is an allowable error in striking each integer precisely. The allowance ap pears, (from music), to be approximately 0.0002 per unit. As values approach 5000 the cumulative error can be one unit, (5000 x 0.0002 = 1 unit). At present we can only experiment with this as we explore the Music of The Spheres. (The two ten thousandths is only approximate as presently de termined.) Every piano tuner is familiar with this same major problem in tuning a piano, to get
47
Quantum Arithmetic
each octave as precise as possible. To put this in a more visible perspective it compares machining a mechanical part to a tolerance of two ten thousandths of a centimeter for each centimeter of thickness.
THE FIRST WATERFALL
To illustrate how the flexibility zone operates, the simplest example is given: Say that three waves are joined in harmony which have values of 3002, 3999, and 5002. They will give nearly per fect harmony because they are near the 3-4-5 ra tios. But they can use the flexibility zone to read just their actual values to true 3, 4, 5 values. This will create a cascade to the Harmonic Cycle which was pictured in the previous chapter. That is to say, they adopt a new unit of measure which is 1000:1, like our change from meters to kilome ters.
Of course, between 5000 and 10,000 there will be cases where, say, not all values could revert to lower values. This dropping in scale is called a "cascading" in a "waterfall" of energy. As an addi tional problem, there will usually be from four to seven such waves joining in harmony. All of them may not cascade at the same exact time. If some cascade at slightly different times, this brings in the "phasing-in" features which was explained earlier.
There may be as many as seven or eight cas cades, or waterfalls between the creation of the unit energy, Iota, and the period of, say, 0.1 vibra tion per second.
Between the wavelengths of visible light and wavelengths of sound, there appear to be two cas cades. That is why we use Angstroms for measur ing light, and Meters for measuring sound wave lengths. Nature does, essentially the same thing.
ORGANIZING ENERGY
The features described above lead us to the formatting of energy. It is first, assumed, that the shortest energy cycles to be considered are at a frequency of 4 quadrillion cycles per second, or a frequency of 1015 hertz. (This is energy. The Forces which generate the energy are much faster, possi bly up to 1040 hertz.)
The forces generate this energy at a uniform rate of 4 quadrillion hertz, and all units of energy, Iota, are presumed to be the same size. These units of energy combine is aggregates of 3, 4, 5, or 7 units, to form their basic waves. (They do not
aggregate in two's). These discrete waves can then join in various combinations to form any one of the four basic Harmonic Cycles. These form the waves which have no factor larger than 7, (except powers of 2, 3, 5, or 7), in their quantum number. This will constitute the first "waterfall" of energy.
We do not find anything to prohibit them , from also joining in larger aggregates, but they probably do not at this first waterfall. There is a possiblility they could join in harmonic cycles which create waves having integers up to 10, in their quantum number. If the Harmonic Cycles having quantum numbers up to 10 can be aggre gated, that would be the limit because 7 x 8 x 9 x 10 = 5040. In this case there could be only fifteen different Harmonic Cycles. (Check it out!).
Of these, there would be 7 female Harmonic Cycles and 7 male Harmonic Cycles, and the ba sic 3, 4, 5, cycle based on the unit quantum num ber of 1, 1, 2, 3. There would be attraction be tween any male and any female Harmonic Cycle. But there would be repulsion between male cycles and between female cycles. This is the beginning of Music Of The Spheres.
All of these notes are "good", but when they join into pairs most of the chords will be "good", but some of them will be “bad”. We say there will be "harmony", or there will be “dischord”. We un derstand the harmony and dis-harmony, but the “dischord” as found in Music of the Spheres" is like no other chord we have ever heard and real ized, as we heard it.
One must hear Music of the Spheres played in order to appreciate the above paragraph. The (good), harmony is achieved through that "Quant um Flexibility", previously described. It produces that tremolo we hear in a singers voice. It is "good", only so long as the tremolo is moderate and under about 6 vibrations per second.
When that tremolo reaches 15 hertz it be comes nerve wracking and when it reaches 20 hertz it can be disasterous. A case in point is the roar we hear from a storm, or from the ocean. This is in a range of 12 to 15 hertz. We feel it more-so than hear it. That is the musical feature which creates the feeling of awe, even from the roar of thunder. In properly tuning a piano, the strings must be adjusted to keep the "beat" be tween strings, under 7 hertz, when adjusting the "thirds", or the "fifths", to complete an octave.
OTHER WATERFALLS (in wavelength)  Part of the above discussion applies only to
48
Quantum Arithmetic
later waterfalls, after the first Harmonic Cycles are formed. As the fifteen Harmonic Cycles join in pairs and then groups they will be above the 5040 limit and reach an area of instability. The instabil ity is brought on by, and is cured by, the allow ance for "Quantum Flexibility". The wavelengths become indefinite somewhere between 5040 and 10,000 units, because they miss the mark of hit ting integers as precisely as they should. This will begin to appear after about 6 octaves, and com plete instability will occur in the range of 10 or 12 octaves from their starting point, from one water fall to the next.
(At this point it is necessary to clarify, wheth er we are speaking in terms of wavelengths or fre quencies. One is the inverse of the other.
Considering numerical values as "above" in frequency, is "below" in wavelengths, and vice ver sa. The "First waterfall" was described in terms of hertz, but the later ones are described in terms of wavelength.) The myriads between waterfalls with which we are most familiar is in the myriad of sound, and music. A second myriad we are also familiar with, is the myriad in which the single oc tave of visible light occurs. Visible light appears to be in the third or fourth octave in that myriad.
We are familiar with these two myriads, light and sound. There are possibly two myriads be tween them. One seems to be the myriad of mat ter, just above the myriad of sound. The other seems to be the myriad of organic matter, above solid matter, and below the myriad of light.
Below the myriad of sound, is still another my riad. This myriad was being described above to ex plain the "First Myriad". We are familiar with light and sound because we have special senses to re ceive them. We have not been familiar with the myriad below music because there is no special sense organ to receive it. It is called the Myriad of "Mentalism" because it seems to act in the same range as the human nervous system. It might also be called the myriad of human information, which will be discussed in Book 4.
THE SYSTEM OF MYRIADS
The way which energy is divided into myriads can be characterized as a stream of water as it flows to the sea. The sea, in this case can be visu alized as a "black hole" which absorbs all energy after it has spent its useful life. The beginning of energy is presumed to begin at some creative mechanism which manipulates the Forces until such time that it creates the fundamental units of energy.
The fundamental energy is created in un counted units, Iota, which vibrate at one common frequency of four quadrillion vibrations per sec ond. Without discussing, or even knowing what these vibrations are or of what they consist, they will simply be called vibrations, or Iota, and they vibrate at this uniform rate. The rate is in accor dance with the origination of the quantum system and they either fit into this system of units, or they actually generate and control the quantum system. A detailed, possible explanation, can be found in: [Ben Iverson, “The Crystal Universe” in “Journal of Sympathetic Vibratory Physics", Delta Spectrum Research, La Junta, CO, (January 1989).]
CRYSTAL UNIVERSE
For the benefit of the reader, a brief review of the above article, is given. The aetheric universe is theorize as consisting of a three dimensional solid of equilateral triangles which are eight units on each edge. They form into larger aggregations in the shapes of the five platonic solids of various magnitudes. (See Pg. 39 & 40, Book 2). Some may even be Penrose crystals. These triangles, large and small, can be divided into two parts by a sin gle line from any vertice to the opposite side, and this line is measured in whole units. In the 8-unit triangle this line is 7 units and divides the oppo site side into 3 units and 5 units. These four inte gers, 3, 5, 7, & 8, as lines represent the four Forc es, which are, yet undefined.
UNIT OF ENERGY
Each unit of energy can be depicted by any of the basic unit figures which were introduced in the elementary Book 1. They can be represented by: The 4-3-5 right triangle. The 1-5-7 segment of the Koenig Series: The 8-unit equilateral triangle: Or by any other figure generated by the quantum number 1, 1, 2, 3.
Each unit of energy is too small to ever be verified empirically within our systems of experi mentation.
AGGREGATION
These small uniform units of energy are pic tured as flowing across a flat plain and gathering together in aggregates of threes, fours, fives, and sevens, and powers and products of these four primes. There will be very few which gather into larger prime numbers of aggregations. The larger primes would be considered normally as “strays” and limited to being aggregations of twenty differ ent prime sizes. (11, 13, 17, 19, 23, 29, 31, etc.
49
Quantum Arithmetic
up to 97). Even at this size they are probably un detectable, except through mathematics. Whenev er they aggregate, they can be said to plunge over a waterfall and become Harmonic Cycles, and the larger ones as Prime Cycles.
After these cycles cascade over the waterfall they have a definite wavelength. Each waterfall has the same procrustean configuration from top to bottom. The procrustean aspects apply only from the top to the bottom of a waterfall. In the plain, (The Myriad), between two waterfalls, a gen eral plan is followed in which quantum flexibility is allowed. The physical results of the aggregating,
and decomposing, which occurs in this plain, can be extrapolated from one myriad to another, but primarily, only in the mathematical context. This is where harmonics, and Synchronous harmonics occurs. They are strictly mathematical but have predictable physical results.
BOOK 4 FOLLOWS
50
Quantum Arithmetic
NEW BEGINNINGS Book 4
The reader probably realizes that these three books of Quantum Arithmetic are only a begin ning. Having read and understood these pages, only opens the door to far greater knowledge which must still be discovered in the time ahead. This beginning will be absolutely necessary for our future advancement. There must be major chang es in our concepts and uses of mathematics as the implication of Quantum Arithmetic is realized. There may also be minor changes in the Quantum Arithmetic, as it is presented here, when it is fi nally adjusted to productive uses.
As Aristototle says in “Metaphysics”:
"The study of truth is partly hard and partly easy. A proof of this is the fact that no one man is able to grasp it adequately. Yet they do not all en tirely fail. Each says something about the nature of the world, and though individually he adds little or nothing to our understanding of it, still from the combination of all, something considerable is accomplished. Hence, as truth seems to be like the door, which the proverb says, no one can fail to hit, in that respect our study of it is easy. But the fact that we can have some notion of it as a whole, but not of the particular part we want, shows it is difficult. Perhaps too, the difficulty is of two kinds and its cause is not so much in the things themselves, as in us. For, as the eyes of bats are to the brightness of daylght, so is the rea son in our soul to the things that by nature are clearest."
Humanity, as a whole has made the same mis understandings over and over. Many of these have, caused us to misunderstand the important meanings which the Greeks and Egyptians, and others have tried to give us. We tend to make a great "to do" over their puzzlement over the irra tionality of the diagonal of a rectangle, when in fact, there was no puzzlement on their part. The puzzlement is our own. This has blinded our eyes and our minds to the true accomplishment which they made.
The true puzzlement of the Greeks and Egyp tians was not the incommeasurability of the diago nal of a rectangle. Any puzzlement came from their difficulty in understanding the literature which lay before them in several ancient and for eign languages, which used words and names of things which they had not yet learned. Aristotle and others before him misunderstood those an
cient foreign passages every bit as much as we tend to misunderstand the Greeks and Egyptians. They could not see that the Earth was not flat, or that the Sun is the center of our solar system, and that it does not travel around the Earth. When an ancient text told of something so simple as gravity there was no acceptance or comprehen sion of it. It really is not simple because, even to day we do not know what gravity is, or what caus es it.
They did make gains in transmitting to us the more ancient knowledge from those now de stroyed texts, as they understood them. Now, we must winnow the wheat from the chaff. Their un derstanding of the ancient texts was incomplete, just as our understanding of the Greeks is incom plete.
GAINS MADE
Aristotle says, there are small gains and there are great gains. In the years to come we will find that Quantum Arithmetic as written here, is only a small gain, although it may seem great at the present time because we must make great chang es. There is so much more beyond this to be dis covered, proven, and learned. Quantum Arithme tic only opens the door. It is a “door that we could not have failed to hit”, eventually. We must now go through this door.
Now the door is open we can begin to see the bigger picture. All begins with energy. Energy is divided into myriads, and myriads are divided into octaves. A part of one myriad is the octave of visi ble light. A whole myriad is the myriad of audible sound which forms another aspect of our being and our knowledge. Many of those aspects have been entirely missed, or only hinted at by an occa sional researcher.
HUMAN SENSES
We have yet to discover what energy really is. After or before we discover what energy really is, we must learn what light is; What sound is; What harmony really is; And what life is. We must de termine how they relate to energy. We have over looked the many "invisible worlds" which are just as real as the world we recognize through our senses. They are not truly invisible. They are only invisible to our limited senses. We must accept, that we know the world which we recognize only
51
Quantum Arithmetic
because our senses are attuned to perception of that single world, or this single dimenson of ener gy. With instrumentation we have located other worlds, or parts of other worlds, which are as real as our own. A Kirlian photograph, and a hologram are real although we cannot sense their subject matter. They are in another real dimension which is different from the one with which we are famil iar.
We have adopted the vain belief that once we can give a name to an object, a thing, or an ac tion, that we understand it. This is far from the truth. These books of Quantum Arithmetic open the door slightly. Through it we find that our sci ence has only scratched the surface.
In many places, our science has scratched the surface, only to cloud what lies beyond, by assum ing that we have discovered a final truth. We have discovered the metric system and say, this is the "final solution" to standardizing measurement. it is only a small step, and it contributes little or nothing. But as Aristotle says, "it adds little or nothing to our understanding of it, still from the combination of all, something considerable is ac complished". In this respect, the metric system has contributed to our understanding.
This has blinded many to the possibilities which nature has to offer, not only in measure ment, but in all those things which we cannot sense directly. Our world, Our universe, and our lives are exceedingly small when we restict our selves to our senses and the additional instru ments which we see fit to devise. These instru ments do extend our senses, but not necessarily our understanding.
Often those instruments do not measure the critical and proper parameters, because we still must guess what the proper parameter may be. Does a thermometer measure energy? It does not. It measures heat and that is not a viable parame ter. We measure wavelengths, but as shown earli er in this book, wavelength is not a primary crite ria. The primary criteria is in the prime factors used in that wavelength, as described earlier in this Book 3.
THE PATH
In the development of Quantum Arithmetic, many wrong turns were taken. A wrong interpre tation or only a slight error in an to interpretation of results, has often required a retracement of steps taken. One has only to read, "Prelude to “Synchronous Harmonics", (1976). The first entry into Synchronous Harmonics is finally written,
(1991) in the previous chapters of Book 3, and it turns out considerably different than was thought in 1976. The "Prelude" was only a prelude.
It was not that the Greeks who were thought to be dumbfounded by incommeasurability. It is our humanity of today which carry that thought, and perpetuate it. The irrational numbers are not an obstacle to nature. They are a necessary part of nature. The Greeks and other ancients tend to imply this although they did not fully believe it themselves.
ESSENCE OF CHANGE
Incommeasurability is the essence of Nature. Without incommeasurability there could be no change. The old writings give vague and often in direct references to this. It is not stated elsewhere in Quantum Arithmetic, but the necessity for in commeasurability, in order to have change, is a conclusion to which any reader will have to even tually arrive. Again, nature has no perfect circle, and no perfectly straight line because that would forestall any further change.
Quantum Arithmetic is only a means to an end. It is not an end in itself. it will be necessary to make changes and improvements in the inter pretations in many cases where the absolute proof is still missing.
The primary reason for these books is to dis seminate the information for those who follow to fill in omissions; To make corrections where need ed; And to continue the extension of this work.
This volume, essentially concludes this series of books in conventional application of Quantum Arithmetic. However, there will be another booklet which reaches over into the metaphysics of which Plato, Aristotle, Josephus, Lucretius and Marcus Aurelias, among many others wrote. Anyone pur suing Quantum Arithmetic must eventually enter this area in order to understand metaphysics in the common phenomena we encounter every day. In Quantum Arithmetic we find the tools for ex ploring into the mysteries of the mind and into bi ological processes. But we must first learn to ap ply Quantum Arithmetic to our supposedly understood features of science. Through learning how application is to be nade, those applications, will nake it possible to go beyond our present knowledge, and extrapolate far beyond anything we can guess.
Is there a Supreme Intelligence? There has to be. Between our intelligence and that Supreme In telligence, there are many stages of intelligence
52
Quantum Arithmetic
greater than ours. Some beings which possess this intermediate intelligence are benevolent, and some of them may be malevolent, if they do, in re ality, exist.
ALIQUOT PARTS
In the previous text were found several new bits of information and truths. Let us now try to put some of them into context to proceed to the next stage.
Aliquot parts is one of those bits. An aliquot part might be called a quantum part of a whole. In this case the whole means the whole number which is the product of the four integers of a quantum number.
Every quantum number must contain the primes 2, 3, and usually a 5 and/or a 7 for three or four of the prime factors. In addition to these four, it will usually contain one, two or three other prime factors, except in the case of 1,1,2,3. A quantum number will generally contain from five to seven prime numbers.
An aliquot part is, (technically), the product of any two prime numbers. In a quantum sense it is the product of all of the prime numbers excepting one of them. For the complete number, that ali quot part is repeated the number of times indicat ed by the one, odd prime number which is left out. Every aliquot part is always a composite, even number, and every composite number, which is divisible by 6, has the potential of being an aliquot part.
Every wave, or cycle of energy is made of its unique aliquot parts. It may be divided into ali quot parts, in different ways, dependent upon which prime number is left out each time. When aliquot parts are the same, it may be aligned with a second wave or cycle which has this same ali quot part. The second wave may appear to be en tirely different from the first. When any two waves have the same aliquot parts they are said to be "Harmonic" to each other.
The lower the value of that aliquot part, the greater will be the harmony.
Either one of a pair of harmonic waves, when divided into aliquot parts in a different way, may align itself with a third wave which has this new form of aliquot part. Again, the first wave, also di vided in still another a different way will be able to align itself with the same aliquot part in the third wave. When this occurs with the 3-way tie, they form a Harmonic Cycle. But it will be so large that
it may, (with a +2 or -2 derived f rom Quantum Flexibility), CASCADE TO A LOWER VALUED HARMONIC CYCLE, and a lower myriad. The scale change should be about 5040 to one.
There are three different, smallest aliquot parts. These are 2 x 3 x 5 = 30; 2 x 3 x 7 = 42; and 2 x 3 x 5 x 7 = 210 in any scale of units.
PRECURSOR ENERGY
The smallest scale of units is supposed to be a wavelength of one Iota. (Four quadrillion hertz, or 4xl015 hertz. This has been a tentatively assumed value for energy. To go less than this value, will put us, [theoretically], in the domain of the basic four forces, and out of the domain of energy, as such.) Theoretically, also the highest integers al lowable as part of a quantum number, is some what less than 100. Beyond this value quantum stability is lost. The absolute highest quantum number should not produce a right triangle with any side greater than 5040. The reason this is the limit is that discrimination between units, de grades considerably in the range between 5040 and 10,000 units.
Studying this from a purely logical point, The differentiation between 999 and 1000 is much less than the differentation between 99 and 100. This is much less, in turn, than between 9 and 10. The greatest differentiation in values is found between I and 2. The difference between having $1 and $2 is much greater than the differentiation of having a million dollars and having only $999,999. This demonstrates the difference be tween strong quantization and weak or non exis tent quantization. It is actually in this latter area in which contemporary science is working. It car ries over, though it should not?. even when the quantum sciences are being considered.
SECONDARY PRECURSORS
Thinking in terms of this, this frequency, how then, do we get to the frequencies of trillions of hertz? (4 trillion hertz, or 4xl012 hertz). How do we get to frequencies down in a range in which we can work? The answer is that this is accom plished entirely through aliquot parts and har monic resonances between aliquot parts. When resonance between groups of waves occurs, then the waves themselves seem to actually become smaller aliquot parts of a new and larger scale of energy.
This change of scale seems to occur some where between wavelengths of 5040 to 10,000 units. These longer waves would then be meas ured in the low prime numbers between 1 and
53
Quantum Arithmetic
100, instead of numbers between 5040 and 10,000. They will combine in the same way that the higher scale combined and will, in turn, create their own harmonics in a myriad below the water fall.
AUDIBLE SCALE
After proceeding through several myriad scale changes, we eventually arrive at the scale of musi cal energy which is audible. That is to say, we ar rive at the scale in which we have the physical ap paratus, our ears, to receive and analyze these vibration.
In that analyzing we divide the myriad into oc taves and redivide the octaves into audible and discernible notes. In studying the numerical ar rangement of music, it is possible to determine what happens within what is assumed be one my riad of scale of energy. This myriad of scale seems to be divided into octaves and each octave then has four major notes which we obtain in playing a bugle. This gives us an idea of the generalities of how a other myriads of scale of energy may be ar ranged. The C, F, A, C of a bugle is an analog of red, yellow, green, blue of visible light.
VISUAL OCTAVE
Rising from this point we go to a higher scale of energy. Other things can be discerned. In this case we go upward to the electromagnetic scale which we call visible light. Here we find we have the apparatus, our eyes, to discern and differen tiate across exactly one octave of energy, from about 700 millimicron, (7000 angstroms), waves of red light to about 350 millimicrons, (3500 ang stroms), of blue light. Between these we have the yellow and green which conform to the bugle notes of sound. The red, yellow, green and blue conform to the bugle notes C, F, A, C through each octave of sound.
It is from these two myriads of scale of energy, which we derive all of our basic knowledge of sci ence and nature. On this basic knowledge we then have extended the rest of our science through in strumentation and somewhat through logic. Both of these were deficient without the firm basis which Quantum Arithmetic now provides. Much progress has been made in understanding science and nature but some magnificent errors and omis sions are found to have occurred.
SCALE OF MATTER
There seems to be one other myriad of scale of energy which we use, but have not recognized
completely. This is the scale in which we exist, -- the scale at which energy creates matter, or the il lusion of matter, to us, - who are also matter.
But humans are somewhat more than just matter. We also have life, consciousness, and spirit which seem to be an entirely different myri ad, or myriads of scale of energy. In searching out the myriad of scale of energy which becomes mat ter, perhaps we can better understand beyond that.
Matter appears to be a single myriad of scale of energy. it consists of considerably less than 100 basic elements. The radioactive elements are elim inated. The radioactive elements are in the upper range of discernment of the scale from 5050 to 10,000. That is probably the reason they are radi oactive and unstable. They are in the range of weak quantum discernment. They will tend to drop to lower levels and higher quantum discern ment through radioactivity, just as radium turns to lead and helium.
But where on the complete scale of energy does matter lie? Logically it seems to lie at a larger scale than that of visible light. Matter can absorb light and must therefore be a larger scale. Plants absorb light and through photosynthesis builds its volume of matter. These plants can be dried and burned to again release this energy as heat and light. So matter should reasonably lie some where between sound and light.
In the upper range of this scale, at the top of the scale of elemental matter, is the range of chemical combination. The elements absorb ener gy from the scale of electromagnetic, heat and light. This forms chemical combinations between elements which is a temporary state. Photo synthesis also lies in this range, -- Myriad of Chemism.
SCALE LIMITS
No myriad of scale is clearly delineated but the lower bound, in the scale of matter, can be de fined. This can be done through quantization of the spectrographic lines of the elements and their electron energy states. This has been carried out to a degree but is yet incomplete. (See Chemistry in Book 3.) Hydrogen and Helium are at the stable bottom of the myriad of matter, and radioactive elements are at the unstable top.
The myriad of scale of audible sound is usual ly delineated to begin at 32 hertz and end at ap proximately 7000 hertz. This is strictly sensorial judgement and may vary between individuals, as
54
Quantum Arithmetic
is also the case in the octave of light, between red and blue. The lower bound of music may be as low as 25 hertz and may be as high as 60 hertz.
MYRIAD OF MENTALISM.
Below the audible scale is another myriad of scale which is just beginning to be investigated.. This is in the approximate range of 1 hertz to 30 hertz. Humans are susceptible to this range. It might be called the mental / emotional range. This range seems to be received into our bodies di rectly into the nervous system through sympa thetic vibration. We unconsciously receive it from the system of "beats" resulting from music. It is an example of the cascading of energy from one myri ad of scale to a lower myriad.
"Beats" are said to result as the difference be tween musical tones, but this is a minimal defini tion. This difference must be a low prime fraction, (with denominator less than 16), of both tones which generate it. That is to say, the difference tone must divide both of the musical tones and must be an aliquot part of both of them.
The physical effects of the myriad of scale from 1 to 30 hertz is directly upon the nervous system. Those tones from one to about 4 hertz are relaxing tones. They are usually pleasurable and will tend to bring on sleep. From 6 to 12 hertz are tones of increasing emotional activity. From 12 to 20 hertz are tones which are generally extreme in activity and are sometimes destructive. From 20 to 30 hertz the influence is indeterminate as is usually the case at the high end of any myriad of scale.
These, 1 to 30 hertz tones, are often received unconsciously, without accompanying musical tones. They are unrecognized, but can create defi nite moods, not only in the individual but in gen eral populations.
It is this which generates the awe from hearing thunder or the roar of an angry ocean. We feel these tones rather than hear them.
The other three senses taste, smell, and touch, have played only minor parts in developing our knowledge of nature. The visual sense has played the major role. This has left us with the concern about matter and things related to matter. We have learned that matter is composed of energy without understanding the much larger role played by energy.
STANDING WAVES
Solitons, or standing waves, have no explana
tion in today's science. It is otherwise in Quantum Arithmetic. Matter is composed entirely of solitons of energy. Electrons are male solitons. They sur round the protons which are female solitons. Har monic Cycles composed in accordance to Problem #3, in Book 1, page 8, will provide 95 different kinds of solitons. This is one Harmonic Cycle for each element of matter, although there are only about 60 different stable, (non-radio active), eIe ments. The table of elements is further divided into octaves of male and female elements.
Being restricted very much to the myriad of scale of matter, we have termed this "Physics", and the rest of the myriad scales as "metaphys ics". We have considered that physics and matter is all that really exists. Reality carries far beyond this. Quantum Arithmetic carries us into these other myriad scales.
WHY AND WHEREFORE
THE ROOTS
This chapter is inserted to assist the reader in understanding and remembering a mental picture of Quantum Arithmetic.
Why are the integers in the quantum number designated as b, e, d, a? It would seem that, since they are the very beginning they should be a, b, c, d.
One must go back to 1976 at the writing of "Prelude to Synchronous Harmonics". At this time The Pythagorean right triangle was pictured as Base "A"; Altitude "B"; and Hypotenuse C. It was thought these were the primary parameters. In addition to this, the mean of the hypotenuse and the altitude was given as D, and the difference be tween D and the hypotenuse, or difference be tween D and the altitude was taken as E. In "Pre lude to Synchronous Harmonics" the "E" was given as the "Order Number" in relation to D.
Much of what is written in these texts of "Quantum Arithmetic" was known at that time. But it was NOT recognized that the sum and dif ference of the Base and Hypotenuse, (being square numbers), completed the primary parame ters. When, (1976), the ellipse and the equilateral triangles came into the picture, it was realized that a uniform system of identities was possible.
So,here was a whole system of mathematics with a garbled set of identity assignments. It was not until about 1980 that I finally decided to straighten out this mess, and reassign the letters of the alphabet. The only identities I could save
55
Quantum Arithmetic
were D, and E. But since they were secondary pa rameters, and their roots were the primary param eters I assigned the square roots of D and E to be "d" and "e".
It was in 1980 that d+e became a, and d-e, be came b, and Fibonacci came into an already as sembled picture of Quantum Arithmetic. All of the basic work had to be revised to reflect this new letter assignment.
PAR TYPES
The importance of the four types of numbers was well known, to me, long before 1965. They were called the 4n; The 4n+1; The 4n-1; And the 4n-2 integers. This last, differentiated the 4n inte gers from the 2n integers. This was cumbersome to use, so a name was assigned. They were first called the 4-numbers, the 5-numbers, the 3- numbers, and the 2-numbers.
Then came another differentiation of integers based upon 3's with plus one and minus one. In addition to this, there is another number group based on a primacy of 5 and the plus, and minus to that, The pentagonal numbers came into play. These are neither described elsewhere, nor used in these texts.
So, here I was, trying to differentiate between twelve kinds of numbers. They were 3n 1, 3n, and 3n+1; The 4n-2, 4n-1, 4n, and 4n+1; And the 5n 2, 5n-1, 5n, 5n+1, and 5n+2. Every integer can be classified in three ways. It can be classified ac cording to its relation to 3 to 4 and to 5. Within the first sixty integers, every integers will come up with a different combined classification for each. For instance, "What are the three types of classifi cations of number 37?". Integer 37 is 36+1, 36+1, and 35+2, for its classification in relation to 3, to 4 and to 5. To repeat, "what is the 3-type classifi cation of 32?". It is 33-1 or 3n-1 for 3;. It is 32+0 or 4n for 4. And it is 30+2 or 5n+2 for 5. 1 re named them 3-tri for the 3n integers, 4-quad, for the 4n integers and 5-pent for the 5n integers.
They became the 2-tri, 3-tri, 4-tri; The 2-quad, 3-quad, 4-quad, and 5-quad; And the 2-pent, 3- pent, 4-pent, 5-pent, 6-pent, and 7-pent integers. Although 3-tri and 5-pent, are not introduced in these books of Quantum Arithmetic, they are still necessary divisions of integers, in addition to the four number types which Euclid introduced in Eu clid VII, Proposition 28.
The name "quad" did not seem to fit, so I tried to go to a different language from the Greek and Latin forms for "four". Hindustani-Urdu counts by
"Ek, Do, Teen, Char, Paunch, Chai, Sath, Aughth, Nos, Dos". I selected the "Char" for "four-ness", and subsequently simplified that to “Par”. It is not changed from that selection which occurred in 1980.
There is the realization that readers may have related "par" as relating to "parity", which is root intended. The evolution of "par", in this case, de velops from "char".
In the mean time, the 2-tri, 3-tri, 4-tri, and the 2- pent to 7-pent, (the tri and pent), integers still stand. In the examples given above: 37 is 4- tri, 5-par, and 7-pent; 32 is 2-tri, 4-par, and 7- pent. No two integers below 60, will have the same combined classification. They will come in handy when one begins to relate Quantum Arithmetic to solids, and particularly the Platonic solids. These combined number types will be seen reflected in the Harmonic Cycle pictured in Book 3. They are also related to the harmonics of division, both in digits of the quotient and the integers of remain ders. They also reflect from the sexigesimal sys tem of the ancient base 60 of counting.
Curiously, it was 5 to 8 years before 1960 that Synchronous Harmonics was derived. And the precursor number types described above, were used to develop Synchronous Harmonics.
In 1961 a patent application was made for a computer which operated on the principles of Synchronous Harmonics. Patent Number 3,157,355 was granted November 17, 1964. For further reading, any interested parties may obtain those patent papers. Construction of a working model was started after the preliminary model, but never completed. It was found that quantum Arithmetic could be handled in the standard bi nary computer, when BASIC language, and time sharing were developed. Program "Quantize" was developed in 1977, with the old identities.
KOENIG SERIES
To understand the Koenig series may be a problem for many. One starts with a square which is 7-units on a side. In a clockwise or counter clockwise direction, a point is located 3-units from each corner. These points are then joined around the square, creating a square inside the original square. This inner square is 5-units on a side, and the cut off areas will all be 4, 3, 5, prime triangles.
Matching the 5-unit sides to new triangles, to complete a rectangle, construct four more trian gles. This will leave a one-unit square at the cen-
56
Quantum Arithmetic
ter. One may start with a larger square which is 17-units on a side and mark off points which are 5-units from the corners. The 13-unit square will fit precisely into this with 12, 5, 13 triangles cut off from the corners. A second set of 12, 5, 17 con structed inside these make a 7-unit square at the center.
Or one may start with a square which is 23 units on a side and take out 8, 15, 17 triangles, again leaving a 7-unit square at the center.
This is a two way split in the Koenig series. But there is generally a 7 way split at each of these outer squares. So there is a choice of five additional sizes for the secondary outer square which leaves a 7-unit square at the center.
OTHER TEXTS
The information in these texts of "Quantum Arithmetic", is the same as information to be found in the three volumes of "Pythagoras And The Quantum World", (1982-1986). The difference is that "Pythagoras And The Quantum World", and the booklet, "Prelude to Synchronous Harmonics", (1976), are written in the order in which informa tion was firmed up with solid proofs. Although Synchronous Harmonics was known at an early date, it could not be explained until much of the rest of Quantum Arithmetic was in place.
All proofs will appear in the volumes of "Py thagoras And The Quantum World". They are not given in "Quantum Arithmetic" because the means of proof will often be quite obvious after the text material is understood. The object of "Quantum Arithmetic" is to highlight the relationships within prime numbers, and then to relate them to our physical world. To have given the proofs here would have cluttered the book, and destroyed the continuity of it. Since this is considered as a text book, it is carried through, much as current mathematics is taught.
In all these respects, it is understood that these volumes of "Quantum Arithmetic" are dense and compact, but continuous.
They will need expanding, by those who are adept at teaching the different stages in the evolu tion of teaching. The first paragraph on page 56 of Book 2 of Quantum Arithmetic is meant to facili tate the necessary expansion of the text material.
Because these books are dense, a reader may go through them rapidly, but it may take as much a five years to glean sufficient information from them to fully understand Quantum Arithmetic, in
its present state of development. However, before that time, an interested reader will become side tracked on a given point and begin to explore in directions which are not described in these texts.
There are so many outreaching arrows in the material, which point to all areas of science and nature, that they were impossible to include. Many of them have been explored, but are not de scribed in the text, except by an occasional hint at where they seemed to be pointing. The Cattle Problem on page 35 of Book 1, is one of these hints. The chapter on "chemistry" in Book 3, car ries the hint one step further.
Every nook and cranny of science and nature is reached through Quantum Arithmetic. it covers all of the valid current schisms of science, con firming most, but invalidating others. But Quan tum Arithmetic adds other possibilities. One of these is the possibility of dividing geology into dis ciplines of Seismic Geology, Vulcan Geology, Hy dro Geology, and Aeolian Geology. These would apply the four categories which the Greeks called, Earth, Fire, Water and Air to geological forma tions.
Sixteen lettered identities have been assigned to cover all basic calculations in Quantum Arith metic. They are permanent assignments. The user will find occasion to need other variables, but these sixteen should remain unchanged.
In the previous texts, the sixteen identities which relate to specific geometric measurements, have definite interrelationships between them. These were assigned to relate the triangles, cir cles, ellipses, etc. in a numerical way. The mathe matical relationships between these identities and their combinations was described. The basic mathematical requirements and results were also discussed, including the requirements proscribed by the prime numbers.
Quantum Arithmetic is not in any sense, an invented mathematics. It is entirely natural. Much of it is derived from a better understanding of an cient mathematics of more than two millennia ago. There is probably much more to be gained by additional study of ancient texts, and particularly the work of Diophantus of Alexandria, and Eu clid's Geometry.
PARAMETERS
So long as the relationship is mathematical, these relationships become absolute truths. But when these truths are called upon to apply to the empirical sciences there can be many variants of
57
the application.
Quantum Arithmetic
this instability and suddenly regain a new stabili ty?
The applications derive from the conventional parameters used in the physical sciences. But where did these empirical parameters come from? Many of the parameters used in empirical science may not be fully derived. For example: The Fibo nacci numbers are used in some cases, but not as the quantum numbers, as they are in Quantum Arithmetic, which they certainly are.
In musical chords, we have the halves, (or oc taves), the thirds, the fifths, and the sevenths, but the specific note combinations are not those frac tions. The major thirds is given as a ratio of 5:4. The major fifth is a ratio of 3:2; And the major sev enth is a ratio of 15:8.
PROGRESS
The correct natural parameters must be devel oped throughout science. Music is one of the ma jor starting points. Quantizing, prime numbers, and understanding Synchronous Harmonics is all a part of this.
Parameters which are accepted in relation to matter may be lacking in completeness. Some of them in connection with energy probably are more accurate. For instance, temperature is not a measurement of energy. Wattage is a measure ment but it is an incomplete parameter. it is hoped they can be developed with the help of, and in line with, Quantum Arithmetic.
In the previous texts, the value of d/e was de veloped in connection with the Quantum points, (Q-points). They seem to lead us to an acceptable parameter. It happens that d/e relates to the ellip ticity of an ellipse. It is the inverse of the conven tional definition of ellipticity, which is e/d. In the quantization process of Program Quantize, the value e/d was used to develop the quantum num bers from available empirical measurements. As d/e is the distance along the major diameter of an ellipse between Quantum points of that ellipse. This feature allowed us to find relativity between true quantum numbers, but possibly not the ulti mate and absolute quantum numbers individual ly. There is a true and natural unit of measure, but this has not been found. The quantum num bers seem to lead us to other natural parameters.
None of the four integers of a quantum num ber, should be greater than 100. To go beyond that leads to instability and chaos. This instability was discussed in previous chapters, along with the myriad, (10,000 units). Also discussed was, what happens after number values pass through
The sixteen assigned identities are not vari ants because these lead to a workable system. But there remains a number of possible variables which may, or may not, come into play. Only when the entire plan is put into place and con firmed in empirical research, will true and faithful parameters be able to fill these identities.
Another feature to be described is the phe nomena of harmonics between two frequencies. This has been proven mathematically, and pictori ally. It was discussed in a previous chapter. It has been tested empirically, and been found to apply as described, but that may not be a complete in terpretation.
Putting this to actual use will be the final test. Any given interpretation cannot be valid until it has been checked by independent means, and then applied to physical realities in all its variant forms.
The mathematical content stands on its own support, and provides the needed support for fu ture practical applications. The future practical application will provide the proof of the mathe matical organization. In these applications, then, much of what is now scientific theory, can be put on the firmer foundation of Quantum Arithmetic. Some of those theories will become practical, sci entific law.
It is astounding that such a mathematics could be discovered at this late date. Probably even more astounding is the fact that we have made the scientific progress which has been ac complished without this very necessary mathe matics. Science has long theorized a presumed Grand Unified Field. By application of Quantum Arithmetic to physical conditions, there is little doubt that this is the basis of that Grand Unified Field.
Quantum Arithmetic is needed, and it is need ed now. Science cannot be permitted to run ram pant. It is working blind and in chaos without Quantum Arithmetic. A few scientists have used Quantum Arithmetic to remove the fogginess of contemporary science. These few are creating amazing wonders which would be impossible, oth erwise.
The recently formed International Keely Socie ty is in the forefront of this research.
ENERGY IS INFORMATION
58
Quantum Arithmetic
Each specific frequency of energy is a specific bit, or byte, of information. Each quantum fre quency within a given span of each myriad of en ergy has a bit of information connected with it.
We can see this in chemistry. In any simple two-element compound, each of those elements must be in a specific quantum state in order to bind together. That is to say, the energy informa tion between the two elements will correspond. The frequencies will correspond. They may not be the same but they will be harmonics of each other to the extent that the information within each ele ment's frequencies will form cohesive information through its aliquot parts.
The ancients also utilized this feature in lin guistics. Four different societies used alphabets in which the letters corresponded to numerical val ues. These were Sanskrit, Aramaic, Hebrew, and Greek. It is possible that the Inca quippas also carried messages in alpha-numeric relationships. We have lost, or should consider that we have lost the coding for this relationship between letters and numbers. However, it does tell us that numer ical energy is specific information, according to its frequency.
The information in a given frequency depends on which myriad of scale that a specific frequency occurs. If it is the myriad of matter, this informa tion pertains to the formative energy in matter. If it appears in the myriad scale of audible sound it pertains to a note of music. if it is in the myriad which contains the octave of visible light it per tains to a specific color.
PARAMETERS
The specific frequency may have the same nu merical value in all of these myriad scales, but our perception is that the message is different in each case. Is it really different? Or have we not used the proper parameters in classifying the informa tion?
It has been commonly accepted that the tem perature of any thing, is some measure of its ener gy. It is a measure of heat, but heat pertains to the two octaves of energy below the octave of visi ble light. In measuring temperature it appears that we are measuring something other than ener gy frequencies. We should be measuring the fre quency and designate the myriad of scale to which that frequency applies.
From ancient mathematical approaches, each myriad of scale, stepping downward, increases by approximately 5000 fold. We can assume this
multiple is very precisely fixed. If that is so, it is findable.
Then beyond this we should find out whether it is actually the frequency which should be meas ured. Perhaps it is only an aliquot part of a fre quency which applies. With that consideration, it would seem that the photon is an aliquot part in the first myriad scale.
So much for temperature. Next, what about the way we measure mass? Mass, of course ap plies only to the myriad scale of matter. This is where e = mc2 comes in. It will give us a lead as to where the proper parameters lie, in order to derive parameters for measuring energy such that they will be compatible with all myriad scales.
THE MESSAGE
In the letters of the alphabet, we expect the message to be in the alphabet. Messages are formed: in the myriad scale of audible sound; in the myriad of our emotions; in the octave of the myriad of light. These myriad scales have been shown to involve specific frequencies in which we can measure cause and effect. These may not be sufficient to work out analogs for determination of proper quantum parameters.
In audible sound, we can project from the my riad scale above audible sound, (ultra sound), to produce a specific frequency in audible sound. This appears to be the only area in which we can presently control, and know we are controlling, the cascade of one myriad scale to a lower myriad. But if we take all of science and consider the proper parameters, perhaps we can discover other points of cascading and combinations of frequen cies to produce the cascading correctly.
SENSORY INPUT
Energy is information which we can glean di rectly through our sense of sight. Each cone and rod in the retina becomes a pixel of information. The cones seem adapted to color discrimination, which is frequency discrimination. The rods seem designed only to discriminate whether there is en ergy and how much amplitude it carries. The cones predominate near the center of the retina. The rods predominate away from that center. Col or vision is therefore sharpest at the center of vi sion, and there is very little color discrimination at the periphery.
Energy is also information which we can ob tain directly through our sense of hearing. Each hair or fiber in the cochlea of the ear is attuned to obtaining certain frequencies, or more possibly
59
Quantum Arithmetic
the aliquot parts of a frequency, from which we derive a judgement of harmony and harmonics. The aliquot parts designate the difference between a wood wind, a string and a horn, which produce the same frequencies but different wave configura tions. The ear discriminates to that extent. Unlike our eyes, our ears discriminate over a much wider range. the eyes cover only one octave. The sense of hearing covers enough octaves to be a myriad of energy.
The sense of touch appears to stretch over sev eral myriads of energy, It covers heat sensation, which is in the two octaves below light. it covers vibratory sensation which is in the myriad of sound, and in the myriad below that. It covers the sensation of pain which appears to be a much higher myriad.
In this analysis, then, what areas or myriads do the sense of taste and smell occupy? It has been theorized that smell is a fitting physical shapes, into a cavity to which they conform. Could it riot be that it is, indeed, a vibratory discrimina tion that is being sensed?
INFORMATION
There can be little doubt that energy is infor mation. The information seems to come from: Its wavelengths; From the aliquot parts of each wave length; From an ability to correlate between wave lengths; And between aliquot parts. Amplitude of energy seems not to carry the information of the message. It carries only the urgency of the mes sage.
Energy also carries messages outside our sen sory capability. There are messages which pass between different atoms and molecules of matter. This creates the forming and the unforming of chemical compounds, and forming the shapes of matter. Each prime factor of an energy is a bit. Each aliquot part and/or wavelength is a byte of information.
Music Of The Spheres originated long before Pythagoras. "Music of the Spheres", or "Song Ce lestial" is the subject of Archimedes problem "The Cattle Of Thrinacia". In Odyssey of Ulysses they are called "Cattle Of The Sun".
Quantum Arithmetic dictates that the solution consists of eight integers, all below the value of 5,000. The values of the "male" notes are already known integers. The remaining "female" notes will be within 0.0002 per unit of a number which is an aliquot part.
Given those parameters, a solution is impossi
ble to derive with conventional mathematics. It is not impossible with Quantum Arithmetic, but it will be difficult until the full knowledge of this system of mathematics is better understood.
The four "male" notes are 891, 1580, 1602, and 2226. But the first note must be doubled, (raised an octave), to 1782, making them 1580, 1602, 1782, and 2226.
(Notice that these are all factorable by 2, 3, 5 & 7, along with one larger prime num ber between 7 and 89. This is in line with all of Quantum Arithmetic, and indeed, with the chem istry, and astronomy which were demonstrated previously.)
Factors of these four notes are:
1580 = 22, 5 & 79; 1602 = 2, 32 & 89; 1782 = 2, 34 & 11; 2226 = 2, 3, 7 & 53
The female notes used were 754.95383, 1050.7297, 1197.965, and 1547.4254. As 756 = 22x33x7; 1050 = 2x3x52x7; 1197 = 32x7x19; 1548 = 22x32x43. The discrepan cy between the decimal value, and the factored in teger is within the "Flexibility factor" which is de scribed in Book 3. The primary basis of harmony lies in the numbers 2, 3, 5, and 7, and three or fewer larger primes, as factors. These integer val ues for the female notes all factor within the primes 2, 3, 5 & 7, and one larger prime. That is the basis for their inclusion in Music of the spheres.
The values which are given are presumed to be wavelengths. All of the female notes are higher pitched than the male notes. The notes have been given the names: Tui, Li, Sun, Kun, Ken, Kan, Chien, & Chen, in decreasing pitch value. These names are taken from the Book Of Permutations, (I-Ching). In I Ching, Tui, the youngest daughter, represents Lake, or Joyful, and wavelength 756. Li, the second daughter, rep resents Fire, or Clinging, at 1050. SUN represents Wind, or Gentle for the eldest daughter, and wave length of 1197. Kun, the Mother, represents Earth, or responsive Mother, at 1548. These are the four females, but Sun and Tui may possibly be reversed. This order is just a guess. The sons do certainly occur in order of their birth. The youngest, Ken represents Mountain, or Immova ble, at a value of 1580. The second son, Kan rep resents Water, or Dangerous, at 1602. The next is the Father, or the Yellow Bull according to Archi medes. At wavelength of 1782, He represents Heaven, or Activity. This is the primary keynote of Music Of The Spheres. Finally is the eldest son,
60
Quantum Arithmetic
Chen, at a wavelength of 2226 units. Chen repre sents Thunder, or Arousing and perfectly corre sponds with this lowest note. These names for the notes are taken from I-Ching, but their values are taken from Archimedes Cattle Problem.
While Chien is the primary keynote, each of the others is a secondary keynote in one of the lesser keys. To each of these eight keynotes will be added a fractional part of itself. The fractional parts will be 1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 7/7, 1/6, 2/ 6, 3/6, 4/6, 5/6, 1/5, 2/5, 3/5, 4/5, 1/4 & 3/4. The addition of the 7/7 is, of course the octave for each keynote. This will make 144 notes. The normal playing range will be approximately 4 octaves or 30 to 40 notes chosen for the particular musical scale desired.
All 144 notes can be used, and every note will harmonize, or be somewhat enharmonic, with eve ry other note except those immediately adjacent to it. Some of them will be so close together that only a trained ear can discriminate between them. This gives us an idea of how some of the various modes such as Lydian, Ionian, and others, came into be ing. Each mode has a different emotional impact.
A prominent characteristic of these sets of notes is the vibrato which occurs between them, as a function of the 2, 3, 5, 7 prime factors of each note. This vibrato carries an intense emotional message, and should not be used carelessly.
PLATONIC SOLIDS
All of the previous research has focused on Quantum Arithmetic in the linear and plane di mensions. It will be possible to enter into the di mension of solids, and determine how they are re lated mathematically.
The primes 2, 3, and 5 play a prominent part in the formation of the Platonic solids. Note first, that these solids have faces which have three, four, or five edges. There are three which have equilateral triangles. These are the tetrahedron, the octahedron, and the Icosahedron. Between these lie the cube with its square faces and the dodecahedron with its pentagonal faces.
The number of edges on any platonic solid is equal to the number of edges per face multiplied by half the number of faces. It may even be an in dication of how wavelengths of energy can make standing waves which are substantial enough to constitute matter, and particularly crystalline matter.
The pentagon is prepared to reveal much to
us. It is tied closely to the Golden Ratio, and of course the pentagonal shapes are the basis of the Penrose tilings made of darts and kites. Some very special metallic compounds have also been found, which crystallizes into this pentagonal configura tion.
Each of these Platonic solids can be inset in any other, given the proper dimen sions. All five can be inset, one within another. That has been done.
ENDING
These are the major developments of Quantum Arithmetic and its DYNAMIC process, Synchronous Harmonics, in hand. To have read these books and worked the given problems, still does not enlighten one in the immensity of chang es, yet to be made in order to incorporate these into our daily operations. Many of the errors in our present concepts have been found and dem onstrated. These errors, and omissions must be corrected before serious progress can be made.
This is a process which must be accomplished over a period of time, and that peri od of time, may be many years in coming. But in the application of the information in these pages to daily problems, it will be found that the road is made easier to travel.
CONCLUSIONS
Every scientist considers mathe matics a solid, unchanging science. But a new arithmetic has been found which changes all that. Those who ignore it will be left behind.
This new math is not just a re hash of mathematics. It is entirely new, from the foundation up. It is mostly a recovery of mathe matics which was practiced thousands of years ago. It is entirely new to us, but was familiar to the Greek, Egyptian, Arabic, and Oriental philoso phers.
DIFFERENT
What can we do with it? And how is it different? It is different because it ties in pre cisely with quantum sciences, and “quantum" means only “measurement in only whole num bers.” It is different from conventional mathemat ics in that it is absolutely accurate, and it uses only multiplication, and addition of numbers. But with this reduced agenda it is possible to do things which conventional mathematics, with trig onometry and calculus, finds impossible.
61
USES
Quantum Arithmetic
books.
WHERE HAVE WE BEEN
One of the most important things it can do is to find the natural quantum number for anything for which sufficiently accurate empirical measure ments are available. For example, the quantum number for Earth is “60". Then as the Greeks once did, we add an integer, and subtract the same in teger from it, for the other two integers of the quantum number. This gives us a list of four numbers, including the integer which was added and subtracted. In this case, the integer is "1". it results in 59, 1, 60 & 61 for a Fibonacci configu ration for Earth.
Once these four numbers are obtained, one can calculate any relationship between Earth and the Sun, ignoring the effects of any other planet, or even the Moon. This latter is important because the Moon throws the Earth 3000 miles out of its orbit twice every month. At Full Moon we are 3000 miles inside our natural orbit and at New Moon we are 3000 miles outside.
The Greeks knew about Fibonacci numbers more than 1500 years before Fibonacci was born. Euclid described them in Book VII, Proposition 28, in our present geometry books. Even the Chal deans knew. Fibonacci had only a part of the story behind these very important number groups.
With this new math, we can outline nearly all of the paths within, and along which, an atom can travel. In an atom of chlorine there are at least eighty such paths, and Quantum Arithmetic can describe them all. It does this by taking the lines of a spectrograph and finding the precise ellipse, which represents that wavelength of light.
The same thing was done with sodium which has more than 40 paths. It was found that certain paths for a sodium electron was very similar (har monic), to a path for an electron in the chlorine atom. It could be thought that an electron from ei ther atom could follow both paths, and thus, tie sodium and chlorine together to make salt.
These so called, "Fibonacci" numbers can do all that. Laboratory proofs will eventually vindi cate Quantum Arithmetic. In the meantime we can go on and: Do chemistry mathematically; De sign more perfect musical tones and scales; Better understand the cosmos; And perhaps discover a new source of energy, as well as find more efficient applications in all science and technology. Quan tum Arithmetic is here to stay. It will take time to adapt to it and make the modifications which will be required in the basics described in these
Text book number One described the beginnings of this; precise mathematics. Book number 2 carried that on and began to expand on its possible application. These Books (3 & 4), be gin to expand on possible application of this mathematics and its relationships to new and more empirical utilization.
Quantum Arithmetic brings in and, from its different perspective, takes a new look at such old things as prime Pythagorean tri angles; Pythagorean triples which apply to equi lateral triangles, Babylonian unit fractions; Trun cated pyramids; A special kind of ellipse; Equal area quantum circles called Koenig Series; and ties them all together. It also brings in new things to all quantum sciences, music; DNA; and Pen rose crystals.
NEXT
The eventual following book will leave mathematics, and enter the areas which are completely beyond our sensory abilities. it will be based on these four, but will contain no mathe matics. it will enter the area of philosophy, as de rived from mathematical relationships. The Greek mathematicians also used mathematics to develop their philosophy, and they are called Philosophers for that reason.
There is something called Number Theory in conventional math, but Quantum Arith metic does away with much of that. All of its fea tures tie together and makes Mathematical Law out of what is correct in Number Theory. This Law, or these Laws are correct, and proven. (See three volumes of "Pythagoras and the Quantum World"). These Laws go a long way into discarding many things which are incomplete or wrong in mathematics and in science.
NEW FOUNDATIONS FOR THEORY
As described above, Fibonacci's description of his numbers was incomplete. A bet ter description of them is in Euclid. But even Sir Thomas Heath had shortcomings in his commen tary on Euclid, about things he did not under stand. We can go on further with Lord Rayleigh in his origins of wave theory. All of these things are either partly wrong, or are certainly incomplete.
Quantum Arithmetic is a phenom ena. It is surprising that it was not discovered by our civilization two centuries ago. But somewhere
62
Quantum Arithmetic
along the line, science decided it was on the right track. It refused to go back and review its founda tions. By going back and reviewing those founda tions, Quantum Arithmetic was born.
It was born by considering all of those things which were "too trivial to bother with". One, previ ously discarded triviality, stacked upon another eventually found a veritable flood of misinforma tion. Each triviality by itself is insignificant. There is really no new knowledge in Quantum Arithme tic. The only change is the difference in orienta tion, or a difference in the point of view in which the numbers are taken in their relation to each other, and to nature.
DISREMEMBERING
The only difficulty a person will have in com prehending Quantum Arithmetic is their difficulty in changing from, a wrong point of view ingrained in us, to the orientation which has been outlined in these texts. Once a person adopts this changed position of viewing nature, everything falls into place and we find out how wrong we were:
(1) Science has made an error of 3000 miles in the average distance from Earth/Moon orbit to the Sun. Quantum Arithmetic can calculate that down to the nearest tenth of a mile.
(2) Science has developed many disciplines within itself. Quantum Arithmetic tells us that it is possi ble to extrapolate from any discipline to any other. If the first discipline is fully understood, then all science becomes one.
(3) Science has invented rigid units of measure with the thought that they would apply to nature. They do not. Nature has its own rigidly estab lished units of measure, and until we discover one of them we will not know any of them.
(4) Science has no idea of where the Moon is, with any exactitude. The Moon appears to be on a dou ble elliptical orbit, (a torus-like lissajou). One el lipse is in relation to its distance from Earth and one is at approximate right angles to that. With study, Quantum Arithmetic can derive that an swer, but it needs more accurate empirical data from empirical science.
(5) Our civilization is running out of energy, and is beginning to pollute the Earth with the energy it has. But science does not know what energy or vi bration is, in its basic definition. The picture of an atom is due for a drastic change. Quantum Arith metic can begin to pick out the errors in current energy theory, and will eventually come up with a
better understanding , and probably, new sustain able energy sources.
USE OF THESE BOOKS
These books are designed to get the basic information on Quantum Arithmetic, across to those who will want to use it. It will be necessary to expand, and adapt the material, for school instruction, to accommodate specific class needs. That opportunity is offered to anyone will ing to accept that challenge. Free use of this ma terial is permitted for non profit utilization. The author does not intend to profit from this materi al, even though nearly 50 years, and thirty thou sand dollars, has gone into its development. (Re covery of those costs is a consideration.)
REBUILDING THE FUTURE
The material given in these books is so basic, and so radically different, that it will cause consternation to a contemporary Ph.D.., and in scientific and educational institutions. It eradicates many of our beloved concepts, but it does riot invalidate that knowledge which is accu rate. The orientation is changed. This change in orientation, and elimination of previous errors, will allow science and education to progress be yond anything which we have ever considered as a goal.
In the development of Quantum Arithmetic, there have been many wrong paths and sidetracks, which had to be retraced. Many errors in progress were made and some of those errors may remain in the material as given. In the interest of getting this information out, some fea tures have been glossed over by eliminating small er detail.
PREVIOUS BOOKS
Some of these details are given in three volumes of “Pythagoras and the Quantum World” (1982-1986), telling of the development of Quantum Arithmetic as it actually occurred. All required proofs are given in detail in those vol umes.
I conclude here, to expand on the "Cattle Problem" puzzle (Pg. 35, of Book 1) of these textbooks. When readers are able to solve this problem, they will begin to have an under standing of Quantum Arithmetic. The language of the problem is precise. The parameters given in the problem are precise. There is enough informa tion in Books 1 & 2 to solve it, but the reader will
63
Quantum Arithmetic
have to do certain suggested text problems, in full,
to gain the required understanding. It is a real problem and the reward offer will stand until the answer is released. The answer is in integers and the one to solve it will find it so utterly simple that he or she will marvel at its simplicity. It gives the solver every opportunity to err by trying to incor porate errors from conventional mathematics, which they were taught. This problem was first published in 1982 under the title, "Plato's Disks of Gold". It displays how numbers, and particularly prime numbers, fit together to create the world about us in atoms and even galaxies.
SUGGESTIONS
To better understand Quantum Arithmetic, one must work with it arid concentrate on results. it is profitable in convincing one's own mind that Quantum Arithmetic is real, and that some of the present concepts fall short of reality. Only 40 hours of such work, done in short segments of one to four hour stretches, will ingrain it in the memory.
Suggested work is:
(1) Work out a list, of 1000 or more, (prime, [b, e, d, a]) quantum numbers, with all integers below 97. See Book 1, Page 8.
(2) Use all of these quantum numbers to project: Prime Pythagorean triangles; Quantum ellipses; into Koenig Series; Into quantum equilateral tri angles, etc. (Bk. 1, Pg. 40)
(3) Learn to project and extend quantum numbers into (b, e, d, (a), e, d, a) Use these to project exten sions of the above for relations between each fig ure in pairs. (Bk 1, Pg. 27)
These results should be written up for further study and future reference and placed
64
Quantum Arithmetic GLOSSARY
ANCIENT: Artifacts from times preceding Greek history. Older than 2,700 years, and considered legendary.
ENERGY: Undefined in this text, supposedly created by concerted action by the "forces".
FLATLANDER: Reference to the Book "Flat land", describing life in two dimensions.
FORCES: Theorized to be four in number, and each having a positive and negative influence. Each force has a given obligation for the creation of its part of an Iota. Forces also control the aggre gation and flow of energy. Three of the forces work at right angles to each other and may be de picted as latitude, longitude and radial on a sphere. Three forces are tentatively identified as electric, magnetic, gravity, respectively. There is a fourth force which may be spiritual.
IOTA: 'The smallest part possible, It is the original Greek consideration of a point on a line. In this text it is assumed to be a wavelength of a 4-quadrillion hertz vibration.
LIBRARY OF ALEXANDRIA: The library es- tablished by Alexander, "The Great", in Alexandria Egypt, after 329 B.C. (Destroyed 50 B.C. to 640 A.D.). it was a collection of approximately one million ancient writings into one place for refer ence.
MYRIAD: From the Greek terms for ten thousand. Used to segment the energy spectrum in groups of seven to twelve octaves, with sameness, but differ ences between myriads.
OCTAVE: The span of numbers from any integer to the double of that integer.
PAR-VALUE: Classification of all integers in re- lation to the nearest 4n integer. They are the 2-par, (4n+2, or even-odd); The 3-par, (4n+3, or odd even), the 4-par, (4n, or even-even); and the 5-par, (4n+1) or odd-odd) integers. It comes from "char", Hindustani for "four". Even-even, even-odd, odd even, & odd-odd, are Euclid's terminology, Book VII. Par is root parity exception the consideration
that it is a double-parity classification.
PROOF: Proof is used in its mathematical con cept. Empirical "proof" is only corroborative evi dence that interpretations for application are cor rect.
QUANTIZE: A mathematical procedure for deter mining the prime ratio between any two numerical Values. it is necessarily a computer procedure. (See Book #2, page 19).
QUANTUM: Composed of, or dependent upon, whole integers only.
QUANTUM FIGURE: Any geometric figure for which the measurements of its elliptical equivalent will have a square integer for the mean of the peri gee and apogee.
QUANTUM NUMBER: a series of four integers, such that the first added to the second equals the third, and the second added to the third equals the fourth. They are given in the text, as b, e, d, & a. (See Euclid VII, Prop.28)
TREMOLO: A slower vibration, of 0.2 hertz to 30 hertz, imposed on a higher pitched tone. Also called reverberation, or vibrato.
TRIVIALITY: Various features and relationships in mathematics which conventional mathematics has considered inconsequential.
VIBRATION: Undefined in this text, except as a pulsation of energy, created by the "forces".
WAVE PACKET: Periodic points in a composite wave where the composite reaches a maximum, positive or negative value, or where the composite is a periodic null value.
65