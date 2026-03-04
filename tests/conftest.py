"""Shared pytest fixtures for the Spark CSV pipeline test suite."""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Generator

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark() -> Generator[SparkSession, None, None]:
    """Create a local Spark session reused across the entire test run.

    Uses ``local[2]`` so tests work both on a developer laptop and in CI
    without requiring a Docker-based Spark cluster.
    """
    session = (
        SparkSession.builder.master("local[2]")
        .appName("pipeline-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "localhost")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture()
def tmp_output_dir() -> Generator[str, None, None]:
    """Provide a temporary directory for test outputs, cleaned up after use."""
    path = tempfile.mkdtemp(prefix="spark_test_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def sample_df(spark: SparkSession):
    """Return a 50-row DataFrame that mirrors the raw ``dataset.csv`` schema.

    Includes intentional edge-case rows:
    * rows with ``None`` country  (should be dropped by cleaning)
    * rows with ``None`` release_year  (should be dropped)
    * rows with release_year < 2015  (should be dropped)
    * rows with very short descriptions  (may be filtered by token pipeline)
    """
    rows = [
        # --- 40 normal rows ---
        ("s1", "Movie", "The Great Adventure", "United States", "2020", "PG-13",
         "September 1, 2020", "120 min", "Dramas, Action",
         "A brave hero embarks on a dangerous journey to rescue a lost friend from the enemy."),
        ("s2", "TV Show", "Mystery Lane", "United Kingdom", "2021", "TV-MA",
         "January 15, 2021", "1 Season", "Crime TV Shows",
         "A detective uncovers dark secrets in a quiet village while investigating a murder."),
        ("s3", "Movie", "Love in Paris", "France", "2019", "PG",
         "February 14, 2019", "95 min", "Romantic Movies",
         "Two strangers find love during a magical weekend in the city of lights and dreams."),
        ("s4", "TV Show", "Code Zero", "United States", "2022", "TV-14",
         "March 5, 2022", "2 Seasons", "Sci-Fi TV",
         "Hackers fight a rogue AI that threatens the global network and technology infrastructure."),
        ("s5", "Movie", "Desert Storm", "India", "2020", "R",
         "June 20, 2020", "130 min", "Action, War",
         "Soldiers battle through a violent sandstorm to complete a dangerous rescue mission behind enemy lines."),
        ("s6", "Movie", "Happy Days", "Canada", "2021", "G",
         "April 10, 2021", "88 min", "Children, Family",
         "A family celebrates joy and laughter during a fun summer adventure with friends in the countryside."),
        ("s7", "TV Show", "The Dark Hour", "Australia", "2020", "TV-MA",
         "July 8, 2020", "3 Seasons", "Thrillers",
         "A journalist risks everything to expose a scandal involving fear, pain, and betrayal."),
        ("s8", "Movie", "Ocean Wonders", "Japan", "2018", "PG",
         "December 1, 2018", "100 min", "Documentaries",
         "Explore the magical underwater world and discover the wonder of marine life and peace."),
        ("s9", "TV Show", "Laugh Track", "United States", "2021", "TV-PG",
         "August 22, 2021", "1 Season", "Comedies",
         "Friends navigate the chaos of city life with humor, charm, and endless good times together."),
        ("s10", "Movie", "War of Empires", "Germany", "2019", "R",
         "November 30, 2019", "145 min", "Action, War",
         "An epic battle unfolds as empires fight for glory, revenge, and dominance in a dark era."),
        ("s11", "Movie", "Dream Catcher", "Brazil", "2020", "PG-13",
         "May 5, 2020", "110 min", "Dramas",
         "A young woman discovers hope and courage through a journey of self-discovery and adventure."),
        ("s12", "TV Show", "Crime Busters", "United States", "2022", "TV-14",
         "October 3, 2022", "2 Seasons", "Crime TV Shows",
         "Elite detectives solve dangerous crimes with courage, delight, and a passion for justice."),
        ("s13", "Movie", "Space Quest", "United Kingdom", "2021", "PG-13",
         "March 18, 2021", "125 min", "Sci-Fi",
         "Astronauts embark on an adventure to explore a distant planet filled with wonder and danger."),
        ("s14", "Movie", "The Smile", "South Korea", "2020", "PG",
         "September 10, 2020", "92 min", "Romantic Movies",
         "A heartwarming tale of love, kindness, and the magic of a simple smile that inspires hope."),
        ("s15", "TV Show", "Revenge Plot", "Mexico", "2019", "TV-MA",
         "April 25, 2019", "3 Seasons", "Thrillers",
         "A victim of betrayal seeks bloody revenge against the evil crime lord who killed her family."),
        ("s16", "Movie", "Mountain High", "Nepal", "2021", "PG",
         "July 15, 2021", "105 min", "Documentaries, Adventure",
         "Climbers face risk and danger in a brave attempt to conquer the tallest mountain on earth."),
        ("s17", "TV Show", "Family Ties", "United States", "2020", "TV-G",
         "November 1, 2020", "1 Season", "Kids TV",
         "A family shares joy, fun, and warm moments while navigating the challenges of everyday life."),
        ("s18", "Movie", "Blood Moon", "Spain", "2022", "R",
         "January 28, 2022", "115 min", "Horror Movies",
         "A village is terrorized by a dark evil creature that kills under the blood moon night."),
        ("s19", "Movie", "The Hero Returns", "United States", "2020", "PG-13",
         "August 14, 2020", "108 min", "Action",
         "A retired hero returns to battle a new enemy and fight for glory and peace once more."),
        ("s20", "TV Show", "Garden Magic", "Italy", "2021", "TV-G",
         "June 1, 2021", "2 Seasons", "Reality TV",
         "Contestants celebrate the charm and wonder of creating magical gardens filled with delight."),
        ("s21", "Movie", "Silent Fear", "Argentina", "2019", "R",
         "March 12, 2019", "99 min", "Horror Movies",
         "A terrifying force of evil brings death and fear to a remote cabin in the dark woods."),
        ("s22", "TV Show", "Victory Road", "Nigeria", "2022", "TV-14",
         "February 8, 2022", "1 Season", "Dramas",
         "Athletes dream of success and glory while battling personal pain and wound from the past."),
        ("s23", "Movie", "Comic Relief", "United States", "2020", "PG",
         "May 20, 2020", "90 min", "Comedies, Family",
         "A struggling comedian finds fun, hope, and laughter while traveling across the country."),
        ("s24", "Movie", "Night Watch", "Russia", "2021", "R",
         "October 18, 2021", "118 min", "Thrillers",
         "A murder investigation leads a detective into a world of danger, fear, and dark betrayal."),
        ("s25", "TV Show", "Tech World", "Japan", "2020", "TV-PG",
         "September 5, 2020", "2 Seasons", "Sci-Fi TV, Documentaries",
         "Explore how technology transforms our world with wonder, comfort, and magical innovations."),
        ("s26", "Movie", "Sun Valley", "South Africa", "2019", "PG",
         "July 7, 2019", "85 min", "Dramas, Family",
         "A family rebuilds after loss and tragedy, discovering peace and joy in a sunny valley."),
        ("s27", "Movie", "Rescue Mission", "Canada", "2021", "PG-13",
         "December 20, 2021", "112 min", "Action, Adventure",
         "A brave team risks everything in a daring rescue mission through enemy territory and war zones."),
        ("s28", "TV Show", "Cooking Magic", "France", "2022", "TV-G",
         "April 1, 2022", "1 Season", "Reality TV",
         "Chefs celebrate the delight and charm of cooking while competing for glory and success."),
        ("s29", "Movie", "Deep Blue", "Australia", "2020", "PG",
         "August 30, 2020", "97 min", "Documentaries",
         "A deep dive into the ocean reveals wonder, magic, and the beauty of underwater peace."),
        ("s30", "TV Show", "Ghost Trail", "United Kingdom", "2021", "TV-MA",
         "November 15, 2021", "2 Seasons", "Horror TV",
         "Investigators face terror and fear as they explore haunted locations filled with dark evil spirits."),
        ("s31", "Movie", "First Light", "India", "2020", "PG-13",
         "January 5, 2020", "102 min", "Dramas, Romance",
         "A couple discovers love and hope during a magical sunrise adventure in the mountains."),
        ("s32", "TV Show", "Spy Games", "Germany", "2019", "TV-14",
         "June 15, 2019", "3 Seasons", "Action TV",
         "Secret agents fight a dangerous enemy while facing betrayal and risk at every turn."),
        ("s33", "Movie", "Wildlife Wonder", "Kenya", "2021", "G",
         "March 22, 2021", "80 min", "Documentaries",
         "A magical journey through the African savanna reveals the wonder and glory of wildlife."),
        ("s34", "Movie", "Street Fighter", "Brazil", "2022", "R",
         "February 18, 2022", "110 min", "Action",
         "A fighter battles through violence, pain, and danger on the dark streets for revenge."),
        ("s35", "TV Show", "Dream House", "United States", "2020", "TV-G",
         "October 10, 2020", "1 Season", "Reality TV",
         "Families dream of comfort and joy as they build magical houses filled with warm charm."),
        ("s36", "Movie", "Ice Age Tales", "Norway", "2021", "PG",
         "April 15, 2021", "95 min", "Children, Adventure",
         "A fun adventure through the ice age with brave heroes facing danger and discovering friends."),
        ("s37", "TV Show", "Hospital Nights", "South Korea", "2020", "TV-MA",
         "September 20, 2020", "2 Seasons", "Dramas",
         "Doctors face pain, loss, and death while finding hope and courage in a busy hospital."),
        ("s38", "Movie", "The Chase", "Mexico", "2019", "R",
         "August 8, 2019", "105 min", "Thrillers, Crime",
         "A thrilling chase through the dark city as a detective pursues a dangerous murder suspect."),
        ("s39", "TV Show", "Music Box", "United States", "2021", "TV-PG",
         "May 12, 2021", "1 Season", "Music, Documentaries",
         "Artists inspire joy and wonder through the magic of music, celebrating delight and passion."),
        ("s40", "Movie", "Final Stand", "United States", "2022", "PG-13",
         "July 4, 2022", "120 min", "Action, War",
         "Heroes make a brave final stand against the enemy in an epic battle for peace and glory."),
        # --- 5 rows with intentional nulls (should be dropped by cleaning) ---
        ("s41", "Movie", "Null Country", None, "2020", "PG",
         "January 1, 2020", "90 min", "Dramas",
         "This movie has no country and should be dropped by the cleaning pipeline."),
        ("s42", "TV Show", "Null Year", "United States", None, "TV-14",
         "February 1, 2020", "1 Season", "Comedies",
         "This show has no release year and should fail the cleaning filter step."),
        ("s43", "Movie", "Empty Country", "", "2021", "R",
         "March 1, 2021", "100 min", "Horror Movies",
         "This movie has an empty country string and should be dropped by the pipeline."),
        ("s44", "Movie", "Old Movie", "India", "2010", "PG",
         "January 1, 2010", "80 min", "Classic Movies",
         "This movie was released before 2015 and should be filtered out by the cleaning step."),
        ("s45", "Movie", "Ancient Film", "Japan", "2000", "G",
         "June 1, 2000", "70 min", "Classic Movies",
         "This ancient film is too old and should be dropped by the release year filter."),
        # --- 5 rows with short/minimal text (token pipeline edge cases) ---
        ("s46", "Movie", "AB", "France", "2020", "PG",
         "April 1, 2020", "60 min", "Dramas",
         "ok"),
        ("s47", "TV Show", "XY", "Spain", "2021", "TV-PG",
         "May 1, 2021", "1 Season", "Comedies",
         "hi"),
        ("s48", "Movie", "Minimal Text", "Germany", "2022", "PG-13",
         "June 1, 2022", "90 min", "Action",
         "A brave hero fights the evil enemy."),
        ("s49", "TV Show", "Short Desc", "Italy", "2020", "TV-14",
         "July 1, 2020", "2 Seasons", "Thrillers",
         "A dark and sad tale of revenge and danger."),
        ("s50", "Movie", "Last One", "Canada", "2021", "R",
         "August 1, 2021", "95 min", "Dramas, Romance",
         "A magical love story about hope, dreams, and the wonder of finding peace and joy together."),
    ]

    schema = [
        "show_id", "type", "title", "country", "release_year",
        "rating", "date_added", "duration", "listed_in", "description",
    ]
    return spark.createDataFrame(rows, schema=schema)
