
    A Glossary of terms
    Symbol Meaning
    D Database
    ˆD
    Released sample
    N Number of rows
    M Number of columns
    m Size of aux
    X Domain of attributes
    ⊥ Null attribute
    supp(.) Set of non-null attributes in a row/column
    Sim Similarity measure
    Aux Auxiliary information sampler
    aux Auxiliary information
    Score Scoring function
     Sparsity threshold
    δ Sparsity probability
    θ Closeness of de-anonymized record
    ω Probability that de-anonymization succeeds
    r, r' Record
    Π P.d.f over records
    HS Shannon entropy
    H De-anonymization entropy
    φ Eccentricity

    Model
    Database. Define database D to be an N × M matrix
    where each row is a record associated with some individual,
    and the columns are attributes. We are interested
    in databases containing individual preferences or transactions.
    The number of columns thus reflects the total
    number of items in the space we are considering, ranging
    from a few thousand for movies to millions for (say)
    the amazon.com catalog.
    Each attribute (column) can be thought of as a dimension,
    and each individual record as a point in themultidimensional
    attribute space. To keep our analysis general,
    we will not fix the space X from which attributes are
    drawn. They may be boolean (e.g., has this book been
    rated?), integer (e.g., the book’s rating on a 1-10 scale),
    date, or a tuple such as a (rating, date) pair.


    Sparsity and similarity. Preference databases with
    thousands of attributes are necessarily sparse, i.e., each
    individual record contains values only for a small fraction
    of attributes. For example, the shopping history of
    even the most profligate Amazon shopper contains only
    a tiny fraction of all available items. We call these attributes
    non-null; the set of non-null attributes is the support
    of a record (denoted supp(r)). Null attributes are
    denoted ⊥. The support of a column is defined analogously.
    Even though points corresponding to database
    records are very sparse in the attribute space, each record
    may have dozens or hundreds of non-null attributes,
    making the database truly high-dimensional.
    The distribution of per-attribute support sizes is typically
    heavy- or long-tailed, roughly following the power
    law [7, 4]. This means that although the supports of the
    columns corresponding to “unpopular” items are small,
    these items are so numerous that they make up the bulk
    of the non-null entries in the database. Thus, any attempt
    to approximate the database by projecting it down to the
    most common columns is bound to failure.1
    Unlike “quasi-identifiers” [27, 9], there are no attributes
    that can be used directly for de-anonymization.
    In a large database, for any except the rarest attributes,
    there are hundreds of records with the same value of this
    attribute. Therefore, it is not a quasi-identifier. At the
    same time, knowledge that a particular individual has
    a certain attribute value does reveal some information,
    since attribute values and even the mere fact that a given
    attribute is non-null vary from record to record.
    The similarity measure Sim is a function that maps
    a pair of attributes (or more generally, a pair of records)
    to the interval [0, 1]. It captures the intuitive notion of
    two values being “similar.” Typically, Sim on attributes
    will behave like an indicator function. For example, in
    our analysis of the Netflix Prize dataset, Sim outputs 1
    on a pair of movies rated by different subscribers if and
    only if both the ratings and the dates are within a certain
    threshold of each other; it outputs 0 otherwise.
    To define Sim over two records r1, r2, we “generalize”
    the cosine similarity measure:
    Sim(r1, r2) = Sim(r1i, r2i)
    |supp(r1) ∪ supp(r2)|


    Definition 1 (Sparsity) A database D is (, δ)-sparse
    w.r.t. the similarity measure Sim if
    Pr
    r
    [Sim(r, r
    ) >  ∀r
     = r] ≤ δ
    As a real-world example, in fig. 1 we show that the
    Netflix Prize dataset is overwhelmingly sparse. For the
    vast majority of records, there isn’t a single record with
    similarity score over 0.5 in the entire 500,000-record
    dataset, even if we consider only the sets of movies rated
    without taking into account numerical ratings or dates.
    Sanitization and sampling. Database sanitization
    methods include generalization and suppression [26, 9],
    as well as perturbation. The data publisher may only release
    a (possibly non-uniform) sample of the database.
    Our algorithm is designed to work against data that have
    been both anonymized and sanitized.
    If the database is published for collaborative filtering
    or similar data mining purposes (as in the case of
    the Netflix Prize dataset), the “error” introduced by sanitization
    cannot be large, otherwise data utility will be

    lost. We make this precise in our analysis. Our definition
    of privacy breach allows the adversary to identify
    not just his target record, but any record as long as it is
    sufficiently similar (via Sim) to the target and can thus
    be used to determine its attributes with high probability.
    From the viewpoint of our de-anonymization algorithm,
    there is no difference between the perturbation of
    the published records and the imprecision of the adversary’s
    knowledge about his target. In either case, there
    is a small discrepancy between the attribute value(s) in
    the anonymous record and the same value(s) as known
    to the adversary. In the rest of the paper, we treat perturbation
    simply as imprecision of the adversary’s knowledge.
    The algorithmis designed to be robust to the latter.
    Adversary model. We sample record r randomly from
    database D and give auxiliary information or background
    knowledge related to r to the adversary. It is
    restricted to a subset of (possibly imprecise, perturbed,
    or simply incorrect) values of r’s attributes, modeled as
    an arbitrary probabilistic function Aux: XM → XM.
    The attributes given to the adversary may be chosen
    uniformly from the support of r, or according to some
    other rule.2 Given this auxiliary information and an
    anonymized sample ˆD of D, the adversary’s goal is
    to reconstruct attribute values of the entire record r.
    Note that there is no artificial distinction between quasiidentifiers
    and sensitive attributes.
    If the published records are sanitized by adding random
    noise ZS, and the noise used in generating Aux is
    ZA, then the adversary’s task is equivalent to the scenario
    where the data are not perturbed but noise ZS+ZA
    is used in generating Aux. This makes perturbation
    equivalent to imprecision of Aux.
    Privacy breach: formal definitions. What does itmean
    to de-anonymize a record r? The naive answer is to
    find the “right” anonymized record in the public sample
    ˆD
    . This is hard to capture formally, however, because it
    requires assumptions about the data publishing process
    (e.g., what if ˆD contains two copies of every original
    record?). Fundamentally, the adversary’s objective is is
    to learn as much as he can about r’s attributes that he
    doesn’t already know. We give two different (but related)
    formal definitions, because there are two distinct
    scenarios for privacy breaches in large databases.
    The first scenario is automated large-scale deanonymization.
    For every record r about which he has
    some information, the adversary must produce a single
    “prediction” for all attributes of r. An example is the
    attack that inspired k-anonymity [25]: taking the demographic
    data from a voter database as auxiliary information,
    the adversary joins it with the anonymized hospital
    discharge database and uses the resulting combination to
    determine the values of medical attributes for each person
    who appears in both databases.
    Definition 2 A databaseD can be (θ, ω)-deanonymized
    w.r.t. auxiliary information Aux if there exists an algorithm
    A which, on inputs D and Aux(r) where r ← D
    outputs r' such that
    Pr[Sim(r, r
    ) ≥ θ] ≥ ω
    Definition 2 can be interpreted as an amplification of
    background knowledge: the adversary starts with aux =
    Aux(r) which is close to r on a small subset of attributes,
    and uses this to compute r' which is close to r on the
    entire set of attributes. This captures the adversary’s
    ability to gain information about his target record.
    As long he finds some record which is guaranteed to be
    very similar to the target record, i.e., contains the same
    or similar attribute values, privacy breach has occurred.
    If operating on a sample ˆD, the de-anonymization algorithmmust
    also detect whether the target record is part
    of the sample, or has not been released at all. In the following,
    the probability is taken over the randomness of
    the sampling of r from ˆD, Aux and A itself.

    Definition 4 (Entropic de-anonymization) A
    database D can be (θ,H)-deanonymized w.r.t.
    auxiliary information Aux if there exists an algorithm A
    which, on inputs D and Aux(r) where r ← D outputs a
    set of candidate records D and probability distribution
    Π such that
    E[minr∈D,Sim(r,r)≥θHS(Π, r
    )] ≤ H
    This definition measures the minimum Shannon entropy
    of the candidate set of records which are similar to
    the target record. As we will show, in sparse databases
    this set is likely to contain a single record, thus taking
    the minimum is but a syntactic requirement.
    When the minimum is taken over an empty set, we
    define it to be H0 = log2 N, the a priori entropy of
    the target record. This models outputting a random
    record from the entire database when the adversary cannot
    compute a lineup of plausible candidates. Formally,
    the adversary’s algorithm A can be converted into an algorithmA
    , which outputs the mean of two distributions:
    one is the output of A, the other is the uniform distribution
    overD. Observe that forA, the minimum is always
    taken over a non-empty set, and the expectation for A
    differs from that for A by at most 1 bit.
    Chawla et al. [8] give a definition of privacy breach
    via isolation which is similar to ours, but requires a metric
    on attributes, whereas our general similarity measure
    does not naturally lead to a metric (there is no feasible
    way to derive a distance function from it that satisfies
    the triangle inequality). This appears to be essential for
    achieving robustness to completely erroneous attributes
    in the adversary’s auxiliary information.
    4 De-anonymization algorithm
    We start by describing an algorithm template or metaalgorithm.
    The inputs are a sample ˆD of database D
    and auxiliary information aux = Aux(r), r ← D. The
    output is either a record r ∈ ˆD, or a set of candidate
    records and a probability distribution over those records
    (following Definitions 3 and 4, respectively).
    The three main components of the algorithm are the
    scoring function, matching criterion, and record selection.
    The scoring function Score assigns a numerical
    score to each record in ˆD based on how well it matches
    the adversary’s auxiliary information Aux. The matching
    criterion is the algorithm applied by the adversary
    to the set of scores to determine if there is a match. Finally,
    record selection selects one “best-guess” record
    or a probability distribution, if needed.
    1. Compute Score(aux, r) for each r ∈ ˆD .
    2. Apply the matching criterion to the resulting set of
    scores and compute the matching set; if the matching
    set is empty, output ⊥ and exit.
    3. If a “best guess” is required (de-anonymization according
    to Defs. 2 and 3), output r ∈ ˆD with the
    highest score. If a probability distribution over candidate
    records is required (de-anonymization according
    to Def. 4), compute and output some nondecreasing
    distribution based on the scores.
    Algorithm Scoreboard. The following simple instantiation
    of the above template is sufficiently tractable to
    be formally analyzed in the rest of this section.
    • Score(aux, r) = mini∈supp(aux)Sim(auxi, r
    i),
    i.e., the score of a candidate record is determined
    by the least similar attribute between it and the adversary’s
    auxiliary information.
    • The matching set D = {r ∈ ˆD :
    Score(aux, r) > α} for some fixed constant α.
    The matching criterion is that D be nonempty.
    • Probability distribution is uniform on D.
    Algorithm Scoreboard-RH. Algorithm Scoreboard
    is not sufficiently robust for some applications; in particular,
    it fails if any of the attributes in the adversary’s
    auxiliary information are completely incorrect.
    The following algorithm incorporates several heuristics
    which have proved useful in practical analysis (see
    section 5). First, the scoring function gives higher
    weight to statistically rare attributes. Intuitively, if the
    auxiliary information tells the adversary that his target
    has a certain rare attribute, this helps de-anonymization
    much more than the knowledge of a common attribute
    (e.g., it is more useful to know that the target has purchased
    “The Dedalus Book of French Horror” than the
    fact that she purchased a Harry Potter book).
    Second, to improve robustness, the matching criterion
    requires that the top score be significantly above the
    second-best score. This measures how much the identified
    record “stands out” from other candidate records.
    • Score(aux, r)=i∈supp(aux) wt(i)Sim(auxi, r
    i)
    where wt(i) = 1
    log |supp(i)| . 3
    • If a “best guess” is required, compute max =
    max(S), max2 = max2(S) and σ = σ(S) where
    S = {Score(aux, r) : r ∈ ˆD}, i.e., the highest
    and second-highest scores and the standard deviation
    of the scores. If max−max2
    σ < φ, where φ is a
    fixed parameter called the eccentricity, then there
    is no match; otherwise, the matching set consists of
    the record with the highest score.4
    • If entropic de-anonymization is required, output
    distribution Π(r) = c · e
    Score(aux,r
    )
    σ for each r,
    where c is a constant that makes the distribution
    sum up to 1. This weighs each matching record in
    inverse proportion to the likelihood that the match
    in question is a statistical fluke.
    Note that there are two ways in which this algorithm
    can fail to find the correct record. First, an incorrect
    record may be assigned the highest score. Second, the
    correct record may not have a score which is significantly
    higher than the second-highest score.










    We apply
    Algorithm Scoreboard-RH from section 4. The similarity
    measure Sim on attributes is a threshold function:
    Sim returns 1 if and only if the two attribute values are
    within a certain threshold of each other. For movie ratings,
    which in the case of Netflix are on the 1-5 scale,
    we consider the thresholds of 0 (corresponding to exact
    match) and 1, and for the rating dates, 3 days, 14 days,
    or ∞. The latter means that the adversary has no information
    about the date when the movie was rated.
    Some of the attribute values known to the attacker
    may be completely wrong. We say that aux of a record 
    r consists of m movies out of m if |aux| = m, ri
    is non-null for each auxi, and i Sim(auxi, ri) ≥ m.
    We instantiate the scoring function as follows:
    Score(aux, r
    ) = 
    i∈supp(aux)
    wt(i)(e
    ρi
    −ρ
    
    i
    ρ0 + e
    di
    −d
    
    i
    d0 )
    where wt(i) = 1
    log |supp(i)| (|supp(i)| is the number of
    subscribers who have rated movie i), ρi and di are the
    rating and date, respectively, of movie i in the auxiliary
    information, and ρ
    i and d
    i are the rating and date
    in the candidate record r.5 As explained in section 4,
    this scoring function was chosen to favor statistically unlikely
    matches and thus minimize accidental false positives.
    The parameters ρ0 and d0 are 1.5 and 30 days, respectively.
    These were chosen heuristically, as they gave
    the best results in our experiments,6 and used throughout,
    regardless of the amount of noise in Aux. The eccentricity
    parameter was set to φ = 1.5, i.e., the algorithm
    declares there is no match if and only if the difference
    between the highest and the second highest scores
    is no more than 1.5 times the standard deviation. (A constant
    value of φ does not always give the equal error rate,
    but it is a close enough approximation.)