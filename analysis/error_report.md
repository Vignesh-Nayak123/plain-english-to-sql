# Error analysis: finetuned (validation)

Execution accuracy: **667/1034 = 64.5%**  |  failures: 367

Method notes: failure categories come from SQLite error strings and from comparing gold vs predicted SQL text. Difficulty buckets are a heuristic on the gold SQL, not Spider's official hardness. Counts from structural tags can overlap.

## 1. Why predictions fail

| Failure type | Count | Share of failures |
|---|---:|---:|
| valid SQL, wrong result | 204 | 55.6% |
| no such column | 152 | 41.4% |
| no such table | 8 | 2.2% |
| misuse of aggregate | 2 | 0.5% |
| ambiguous column | 1 | 0.3% |

163 of 367 failures (44.4%) do not even execute. These are the cases an execution-feedback (self-correction) loop could in principle fix; the ceiling would be +15.8 points, and a real gain is expected to be smaller.

## 2. Accuracy by approximate difficulty

| Bucket | Correct / total | Accuracy |
|---|---:|---:|
| easy | 272/333 | 81.7% |
| medium | 320/536 | 59.7% |
| hard/extra | 75/165 | 45.5% |

## 3. Accuracy when the gold query uses a feature

| Gold query has | Correct / total | Accuracy |
|---|---:|---:|
| join | 197/408 | 48.3% |
| group by | 161/277 | 58.1% |
| order by | 142/237 | 59.9% |
| subquery | 36/83 | 43.4% |
| set operation | 39/80 | 48.8% |
| having | 43/79 | 54.4% |
| distinct | 39/87 | 44.8% |

## 4. Valid-but-wrong queries (204): structural difference from gold

| Difference | Count |
|---|---:|
| same structure (column/value/condition choice) | 98 |
| missing join | 36 |
| extra join | 30 |
| missing distinct | 17 |
| extra sub | 16 |
| missing sub | 13 |
| missing setop | 11 |
| missing order | 9 |
| missing limit | 9 |
| extra distinct | 8 |
| missing group | 6 |
| extra group | 3 |
| extra order | 2 |
| extra limit | 2 |
| missing having | 2 |
| extra setop | 1 |
| extra having | 1 |

## 5. Weakest databases (20+ questions)

| Database | Correct / total | Accuracy |
|---|---:|---:|
| world_1 | 39/120 | 32.5% |
| car_1 | 39/92 | 42.4% |
| dog_kennels | 40/82 | 48.8% |
| pets_1 | 23/42 | 54.8% |
| student_transcripts_tracking | 44/78 | 56.4% |
| wta_1 | 36/62 | 58.1% |
| course_teach | 21/30 | 70.0% |
| concert_singer | 33/45 | 73.3% |

## 6. Sample failures

**no such column** (concert_singer): Show the name and the release year of the song by the youngest singer.

- gold: `SELECT song_name ,  song_release_year FROM singer ORDER BY age LIMIT 1`
- pred: `SELECT T2.Name , T2.Song_release_year FROM singer AS T1 JOIN singer_in_concert AS T2 ON T1.Singer_ID = T2.Singer_ID WHERE T1.Age = ( SELECT min(Age) FROM singer )`
- error: `no such column: T2.Name`

**no such column** (concert_singer): What are the names and release years for all the songs of the youngest singer?

- gold: `SELECT song_name ,  song_release_year FROM singer ORDER BY age LIMIT 1`
- pred: `SELECT T2.Song_name , T2.Song_release_year FROM singer AS T1 JOIN singer_in_concert AS T2 ON T1.Singer_id = T2.Singer_id WHERE T1.age = ( SELECT min(age) FROM singer )`
- error: `no such column: T2.Song_name`

**wrong result** (concert_singer): How many singers are from each country?

- gold: `SELECT country ,  count(*) FROM singer GROUP BY country`
- pred: `SELECT count(*) , country FROM singer GROUP BY country`

**wrong result** (concert_singer): What is the maximum capacity and the average of all stadiums ?

- gold: `select max(capacity), average from stadium`
- pred: `SELECT max(capacity) , avg(capacity) FROM stadium`

**misuse of aggregate** (car_1): What are the names and ids of all makers with more than 3 models?

- gold: `SELECT T1.FullName ,  T1.Id FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker GROUP BY T1.Id HAVING count(*)  >  3;`
- pred: `SELECT FullName , Id FROM car_makers WHERE COUNT(*) > 3`
- error: `misuse of aggregate function COUNT()`

**no such table** (wta_1): Find the name of tourney that has more than 10 matches.

- gold: `SELECT tourney_name FROM matches GROUP BY tourney_name HAVING count(*)  >  10`
- pred: `SELECT T2.tournament_name FROM MATCHES AS T1 JOIN tournaments AS T2 ON T1.tournament_id = T2.tournament_id GROUP BY T2.tournament_name HAVING count(*) > 10`
- error: `no such table: tournaments`

**no such table** (wta_1): What are the names of tournaments that have more than 10 matches?

- gold: `SELECT tourney_name FROM matches GROUP BY tourney_name HAVING count(*)  >  10`
- pred: `SELECT DISTINCT T2.tournament_name FROM MATCHES AS T1 JOIN tournaments AS T2 ON T1.tournament_id = T2.tournament_id GROUP BY T2.tournament_name HAVING count(*) > 10`
- error: `no such table: tournaments`

**misuse of aggregate** (student_transcripts_tracking): What are the names and ids of every course with less than 2 sections?

- gold: `SELECT T1.course_name ,  T1.course_id FROM Courses AS T1 JOIN Sections AS T2 ON T1.course_id  =  T2.course_id GROUP BY T1.course_id HAVING count(*)  <=  2`
- pred: `SELECT course_name , course_id FROM courses WHERE count(*) < 2`
- error: `misuse of aggregate function count()`

**ambiguous column** (dog_kennels): Which dogs have not cost their owner more than 1000 for treatment ? List the dog names .

- gold: `select name from dogs where dog_id not in ( select dog_id from treatments group by dog_id having sum(cost_of_treatment)  >  1000 )`
- pred: `SELECT name FROM Dogs WHERE dog_id NOT IN ( SELECT dog_id FROM Treatments AS T JOIN Professionals AS P ON T.professional_id = P.professional_id JOIN Dogs AS D ON T.dog_id = D.dog_id WHERE P.home_phone > 1000 )`
- error: `ambiguous column name: dog_id`

