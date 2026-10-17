CREATE TABLE movies (
    id INT AUTO_INCREMENT PRIMARY KEY,
    movie VARCHAR(255),
    country VARCHAR(100) NULL,
    seasons INT NULL,
    star VARCHAR(255) NULL
);

CREATE TABLE movies_tmp (
    movie VARCHAR(255),
    country VARCHAR(100) NULL,
    seasons INT NULL,
    star VARCHAR(255) NULL
);

-- Import the files in GCP Console


INSERT INTO movies (movie, country, seasons, star)
SELECT movie, country, seasons, star
FROM movies_tmp;


DROP TABLE movies_tmp;



-- Clean movie and star names
REPLACE(
  REPLACE(movie, "http://dbpedia.org/resource/", ""),
  "_",
  " "
)

