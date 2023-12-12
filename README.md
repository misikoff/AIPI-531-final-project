# AIPI-531-final-project

Authors: Alonso Guerrero (ag692) & Thomas Misikoff (twm29) 

## Objective

Use a LLM recommender to recommend films to users. Compare the LLM recommendation against a simple baseline recommender and against other modifications.

## Data

## Baseline recommender

The baseline recommender is a recommender with a statitic recommendation: always recommends the "most popular film". Originally, to obtain the most popular film we thought it best to pick the film that was most watched among the users in the dataset. However, the resulting film, "The English Patient", actually never corresponds to the Ground Truth.

Instead, what we opted was to use outside knowledge to hand-pick our baseline film recommendation. The dataset was put together on 1998. At that moment, the highest grossing film, by far, was "Titanic" (1997). Recommending that film always results in a HR of 2/170 = 0.01 (it was actually the ground truth 2 times of out 170).

## Zero-Shot Next-Item Recommendation

Explanation + how we made it more efficient

## Zero-Shot Next-Item Recommendation + Item Features

We attempted to improve the previous algorithm by including item features: years of release and all of their genres. We extracted the data from the `ml-100k/u.item` file. Here's the description of the dataset from its documentation:

Information about the items (movies); this is a tab separated list of movie id | movie title | release date | video release date |IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western | The last 19 fields are the genres, a 1 indicates the movie is of that genre, a 0 indicates it is not; movies can be in several genres at once.

After creating dictionaries summarizing this information, we create a new prompt function: `get_prompt_features()`. The difference with the original `get_prompt()` is that next to each film, we include a parenthesis with the year and its genres. For example, the new prompt would look like this:

*Candidate Set (candidate movies with release year and genres): Die Hard 2 (1990, Action, Thriller), Stargate (1994, Action, Adventure, Sci-Fi), The Crow (None, ), GoldenEye (1995, Action, Adventure, Thriller), Clear and Present Danger (1994, Action, Adventure, Thriller), Waterworld (1995, Action, Adventure), Batman Forever (1995, Action, Adventure, Comedy, Crime), First Knight (1995, Action, Adventure, Drama, Romance), Terminal Velocity (1994, Action), Natural Born Killers (1994, Action, Thriller), Highlander (1986, Action, Adventure), Independence Day (ID4) (1996, Action, Sci-Fi, War), Star Trek IV: The Voyage Home (1986, Action, Adventure, Sci-Fi), Young Guns (1988, Action, Comedy, Western), Days of Thunder (1990, Action, Romance), The Shadow (None, ), In the Line of Fire (1993, Action, Thriller), Under Siege 2: Dark Territory (1995, Action), Money Train (1995, Action).
The movies I have watched (watched movies with release year and genres): Happy Gilmore (1996, Comedy), Boomerang (1992, Comedy, Romance), Made in America (1993, Comedy), Grease 2 (1982, Comedy, Musical, Romance), Michael (1996, Comedy, Romance), City Slickers II: The Legend of Curly's Gold (1994, Comedy, Western), Beverly Hills Cop III (1994, Action, Comedy), Grumpier Old Men (1995, Comedy, Romance

Unfortunately, adding this explicity set of information actually makes the HR slightly worse: 80/170=0.47. The reason might be that the LLM is taking too much into account the explicit genres instead of taking a more holistic appoach to the similarities between films. 

## Instructions

# .py file

```bash
$ python three_stage_0_NIR.py --api_key <api_key>
```

# Notebook
1. Set up environement variables

    ```bash
    $ cp .env.example .env
    ```

2. Add your API key to the `.env` file

3. Start jupyter notebook

    ```bash
    $ jupyter notebook
    ```

