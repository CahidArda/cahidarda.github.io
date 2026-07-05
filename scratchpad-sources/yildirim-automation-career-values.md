# Yildirim, Petrova, Schubert et al., "Automation, Career Values, and Political Preferences"

Status: I verified the numbers via the Wharton/Knowledge@Wharton summary (which quotes them
verbatim), NOT the primary paper.

What I used in the post (prose, Section 2):
- One robot per 1,000 workers cut local average "career value" by ~$3,900 (2004-2008),
  ~$2,480 (2008-2016)  [Wharton summary: confirmed verbatim]
- Automation-hit regions cut long-term investment (housing permits -13%, lower college
  enrollment) and shifted toward populist/GOP candidates in 2016

Paste the primary paper (abstract + key results) below if you want me to confirm against it
rather than the Wharton write-up. Otherwise this one is already solidly sourced.

---

<!-- paste full text here -->
NBER WORKING PAPER SERIES
AUTOMATION, CAREER VALUES, AND POLITICAL PREFERENCES
Maria Petrova
Gregor Schubert
Bledi Taska
Pinar Yildirim
Working Paper 32655
http://www.nber.org/papers/w32655
NATIONAL BUREAU OF ECONOMIC RESEARCH
1050 Massachusetts Avenue
Cambridge, MA 02138
July 2024
Maria Petrova thanks funding from the European Research Council (ERC) under the European
Union Horizon 2020 research and innovation program (Grant Agreement 803506), the Plan
Nacional project PID2020120118GBI0, and the Severo Ochoa Programme for Centres of
Excellence in R&D (Barcelona School of Economics CEX2019-000915-S), funded by MCIN/
AEI/10.13039/501100011033. Yildirim thanks the Mack Institute and the Dean’s Research fund
at the Wharton School for their research support. The views expressed herein are those of the
authors and do not necessarily reflect the views of the National Bureau of Economic Research.
NBER working papers are circulated for discussion and comment purposes. They have not been
peer-reviewed or been subject to the review by the NBER Board of Directors that accompanies
official NBER publications.
© 2024 by Maria Petrova, Gregor Schubert, Bledi Taska, and Pinar Yildirim. All rights reserved.
Short sections of text, not to exceed two paragraphs, may be quoted without explicit permission
provided that full credit, including © notice, is given to the source.
Automation, Career Values, and Political Preferences
Maria Petrova, Gregor Schubert, Bledi Taska, and Pinar Yildirim
NBER Working Paper No. 32655
July 2024
JEL No. J01,L6,M0,M20,M29,M55,O14,O3,P0
ABSTRACT
Career opportunities and expectations shape people’s decisions and can diminish over time. In
this paper, we study the career implications of automation and robotization using a novel data set
of resumes from approximately 16 million individuals from the United States. We calculate the
lifetime "career value" of various occupations, combining (1) the likelihood of future transitions
to other occupations, and (2) the earning potential of these occupations. We first document a
downward trend in the growth of career values in the U.S. between 2000 and 2016. While wage
growth slows down over this time period, the decline in the average career value growth is mainly
due to reduced upward occupational mobility. We find that robotization contributes to the decline
of average local labor market career values. One additional robot per 1000 workers decreased the
average local market career value by $3.9K between 2004 and 2008 and by $2.48K between 2008
and 2016, corresponding to 1.7% and 1.1% of the average career values from the year 2000. In
commuting zones that have been more exposed to robots, the average career value has declined
further between 2000 and 2016. This decline was more pronounced for low-skilled individuals,
with a substantial part of the decline coming from their reduced upward mobility. We document
that other sources of mobility mitigate the negative effects of automation on career values. We
also show that the changes in career values are predictive of investment in long-term outcomes,
such as investment into schooling and housing, and voting for a populist candidate, as proxied by
the vote share of Trump in 2016. We also find further evidence that automation affected both the
demand side and supply side of politics.
Maria Petrova
Universitat Pompeu Fabra
Ramon Trias Fargas 25-27
Barcelona 08005
Spain
petrova.ma@gmail.com
Gregor Schubert
gregor.schubert@anderson.ucla.edu
Bledi Taska
bt540@nyu.edu
Pinar Yildirim
The Wharton School
University of Pennsylvania
3730 Walnut Street
Philadelphia, PA 19104
and NBER
pyild@wharton.upenn.edu
1 Introduction
Inequality is on the rise in many countries around the world. The possibility to move up the income
ladder, or upward mobility, is a way to mitigate the discrepancies in access to opportunities. One of the
defining features of the “American Dream” is that any person who works hard enough can eventually
move to the higher income brackets (Samuel, 2012). At the same time, opportunities for promotions and,
generally, moving up the income ladder might change over time (Chetty et al., 2017; Putnam, 2015).
Economic transformation and, in particular, technological change can affect not only today’s outcomes
but also future occupations and incomes. In this paper, we study how economic shocks affect career
progression and job-related upward mobility as well as the consequences of these changes.
We employ data on job-to-job transitions in the United States from Burning Glass Technologies
(BGT). This BGT data set consists of data on individual resumes available on the Internet from more
than 3000 online job boards, such as LinkedIn, Monster.com, etc., yielding 178 million sequential workeryear observations. We first document that the number of occupational transitions, including transitions
within and between companies, has been declining in 2004-2016 for most occupational groups (see Figure
1).1 Furthermore, we show that, on average, the transitions to higher-income jobs have declined, while
the transitions to lower-income jobs have increased (Figure 2). To figure out what kind of opportunities
holding a particular job at a particular moment in time provides, we would like to have a single measure
that combines information about both future career mobility and the wage potential of occupations.
We propose a measure of expected future income from holding a particular occupation today based on
occupation-to-occupation transitional probabilities and wages corresponding to these occupations. We
call this measure a “career value,” to reflect that it is an expected value of starting a particular career
in a given occupation. Consistent with Figure 1 and Figure 2, career value growth has been declining
over time, especially after 2008 and the Great Recession. We then study how economic changes affect
career values and, thus, expected upward mobility. We analyze how these expected future opportunities
affect individual-level long-term decisions. We also look further at how these changes translate to changes
in economic and political landscapes. We generally follow Acemoglu and Restrepo (2020) to study the
impact of robotization, using industry-level robot adoption in the EU to instrument for potential US
robot adoption.
More specifically, we first document several facts about job-to-job transitions. We note that the
overall number of these transitions has been declining since 2000, particularly so for upward transitions,
i.e., transitions to better-paid jobs or jobs with more responsibilities (management, architecture, and
engineering occupation groups). We then use career value as a single index to capture expected job
transitions and their quality, measured from an income perspective. We define the career value of a
(6-digit) occupation as an average expected income flow corresponding to this occupation, taking into
account the probabilities of transitions from this occupation to future occupations. While doing so, we
weight the transition matrix to make the occupation structure representative of the US population. We
then estimate the transitional probabilities from the resume data and combine them with the wages,
1We an analog of this figure based on CPS in the Appendix.
1
Figure 1: Probabilities of Transition out of 2-digit Occupation Groups, 2000-2015
Notes. For any sector and time period, the probability of transition out of the sector is the number of individuals who work
in that sector at the beginning of the period and whose occupations at the end of the period are different, over the number of
individuals who work in that sector at the beginning of the period. Occupations for individuals are reported based on 2-digit
Standard Occupation Codes and there are 23 unique occupations. Individuals with missing occupational values are dropped
from the calculations. The transitions reported in this figure are calculated over 5 years, where the x-axis corresponds to
the probability for the 2000-2005 period and the y-axis corresponds to the 2010-2015 period.
Figure 2: Transitions to Different Types of Occupations, by Time Period
Notes. The figure depicts the normalized transitions to better-paid, worse-paid, and (approximately) equal-pay jobs over
5-year time periods between 2000 and 2015. Annual job transition probability out of occupation o is calculated as the share
of individuals in the BGT data employed in occupation o in period t and in any other occupation than o in time period
t + 1. Transition to occupation o is calculated by taking all the individuals changing occupations between time period t and
t + 1 in the BGT data, and calculating the share employed in occupation o in period t + 1. Normalization is carried out
by dividing the transition probabilities between specific occupations by the overall probability of transitioning between any
two occupations between period t and t + 1. Occupations held for less than 180 days are excluded. Plotted probabilities are
moving averages over 3 years around the year of interest.
2
corresponding to these occupations, at the state level. We thus use realized occupational transitions
to measure expected occupational transitions and realized wages at the local (commuting zone) and
occupational levels as the best available proxies for the expected wages.
Second, we show that automation and robotization affected both transitions and career values. Higher
robot adoption led to more transitions to similar-wage jobs and fewer transitions to higher-wage jobs, more
so in high-manufacturing commuting zones (i.e., commuting zones with above-median manufacturing
employment). We further note that exposure to robots reduced career value components coming from
both wages and upward transition probabilities. Moreover, although both low- and high-skilled workers
face wage declines at similar rates in response to automation, expected upward job mobility declined
substantially more for low-skilled workers compared to high-skilled workers. These results together point
out towards the decline in middle-level jobs as being a mechanism for the career effects of robotization.
Third, we document several related economic effects that help explain other mechanisms behind the
automation-induced decline in upward mobility. In particular, we find that automation affected not
only jobs in manufacturing but also jobs in other, mostly non-tradable industries, such as services and
retail. We see a related decline in consumer expenditures. In addition, we report a decline in economic
expectations, as measured by Gallup survey data. These findings together suggest the existence of
important spillovers that are responsible for the changes that we document.
Fourth, we find that the negative effects of automation are mitigated in places with higher opportunities for upward mobility. Getting education is the main source of upward mobility, and we observe
this mitigation effect for school enrollment of individuals 18 and over. We also report a similar, albeit weaker, effect for the intergenerational mobility effect of neighborhood characteristics (Chetty and
Hendren, 2018).
Fifth, to understand the implications of observed changes in career values and upward mobility, we
explore the relationship between higher expected career values and long-term investments and political
behavior. We instrument career values using wages and occupational transitions in commuting zones
further than 100 miles away from the focal commuting zone and find positive effects on new home
construction, the share of individuals in higher education, and vote shares in 2016 presidential elections.
Finally, we zoom in on the impact of automation on political outcomes. We document the reducedform relationship between local automation, as predicted by labor market shares in the 1990s, and
industry-level automation in EU countries, and voting for Trump in the 2016 presidential elections. We
find supportive evidence that the effect of automation on voting comes from both the supply side of
politics, as captured by voting patterns of congressional representatives and patterns of campaign visits,
and the demand side of politics, as captured by parallels in automation effects on individual economic
outcomes and voting intentions. We observe a stronger relationship between Donald Trump’s vote share
and the family economic situation of younger respondents, which further supports a career value-based
interpretation.
There are several caveats about our findings. Migration could contribute to our estimates of career
values by changing the occupational composition of workers. However, most US workers don’t move
3
across commuting zones for job related reasons (e.g., Marinescu and Rathelot (2018)). Moreover, our
results suggest that robotization had no effect on the occupational composition. Though in-migration
indeed responds to robotization (Faber et al., 2019), it does not major enough to shift aggregate career
values.
Globalization could be another factor contributing to the decline in career values. Following Acemoglu
and Restrepo (2020), we check that our results are robust to controlling for China shock (Autor et al.,
2013). We do not study China shock as a separate source of variation for the following reason: for
China shock, most variation is cross-sectional for our time period (2000-2016), while our aim is to study
economic shocks over time and the corresponding change in career values. For robotization, time-series
component of variation is at least as important as cross-sectional one.
Heterogeneity of the results could be important to keep in mind while thinking about career implications of robotization. We study heterogeneity for high- vs low-skilled career values. We did not find
any gender-specific differences, either because spillover effects of robotization are large, or BGT gender
identification algorithms are imperfect. We also looked at the heterogeneity for other sources of mobility
(education, intergenerational mobility), and we document that our sources of mobility mitigate the effects
of robotization on career values.
Overall, our results imply that in 2000-2016, there has been a decline in economic opportunities,
i.e., job-related upward mobility, and robot adoption contributed to this decline. Moreover, low-skilled
workers’ career prospects were affected the most. This decline in opportunities made people change their
long-term decisions, such as housing and education. They also change their voting behavior, e.g., vote
more prominently for Trump in the 2016 general elections, while politicians, in turn, change their behavior
to address these new demands.
Our paper contributes to several strands of literature. The literature on upward mobility (Solon,
1999; Black and Devereux, 2011; Chetty et al., 2014, 2016, 2017) consistently reports high levels of
heterogeneity and decline in intergenerational mobility among community zones over time in the U.S.
Putnam (2015) highlights declining opportunities for children. We add to this literature by documenting
the declining rates of job-related upward mobility, as measured by the transitions to better-paid jobs
or career values, in the U.S. in 2000-2016. We, furthermore, document that technological change, in
particular robotization, contributed to this decline. We also find that other sources of upward mobility,
such as more opportunities to get higher education or higher intergenerational mobility, mitigate the
negative effects of robotization.
Second, there is literature on growing income inequality and economic changes that contribute to it,
e.g., trade and globalization (Helpman et al., 2016; Antràs et al., 2017), capital intensity of the economy
(Piketty, 2014; Piketty and Zucman, 2014), colonial origins (Acemoglu et al., 2001; Alvaredo et al., 2021),
or housing expenditures (Dustmann et al., 2021). Our findings suggest that technological change and, in
particular, robotization contributed to the growing inequality and the rising gap between the high- and
the low-skilled by diminishing opportunities for job-based upward mobility.
Third, we contribute to the literature on the economic and political consequences of automation,
4
robotics, and artificial intelligence (AI) on jobs, wages, and society in general (Frank et al., 2019).
Acemoglu and Restrepo (2019) and Furman and Seamans (2019) argue that automation growth could
result in the creation of new jobs and occupations due to increased productivity effects while displacing
some labor at the same time, so a decline in employment may not ex ante be the prediction. Alekseeva
et al. (2021) also find that the demand for AI skills in the labor market has increased dramatically during
the 2010-2019 period, suggesting that some job creation is taking place in response to new technologies.
Battisti et al. (2023) find that automation reduces firm demand for routine jobs relative to the demand
for abstract, task-based jobs. Despite some evidence of creative destruction, Acemoglu and Restrepo
(2020) document that between 1990 and 2007, the adoption of robots resulted in loss of jobs in the
U.S. Arntz et al. (2017) argues that these are overestimates. Graetz and Michaels (2018) running a
study of 17 countries, find evidence for increased productivity growth from higher rates of adoption of
robots. Faber et al. (2019) find that robotization affects migration, and they also, independently of us,
document spillover effects of automation and robotization. Cockriel (2023) finds that historically, artisan
shoemakers lost their life-time earnings as a result of technological change towards factory-based shoe
production. Paolillo et al. (2022) compute job robotization risks and discuss workers’ alternatives. We
contribute to this literature by studying the impact of robot adoption on future career opportunities and
the implications of our forward-looking measure of lifetime income on long-term individual decisions and
other outcomes.
Fourth, we build on a literature in urban and labor economics that shows the importance of considering
wage growth through job mobility when thinking about labor markets (e.g., Topel and Ward, 1992). The
importance of variation in these career dynamics across cities is shown by Glaeser and Mare (2001), who
find that the urban wage premium is driven in part by higher wage growth over time in urban areas, so
the trajectory of wages is important for understanding the attractiveness of different labor markets.
More recently, Bilal and Rossi-Hansberg (2021) develop and test a model where location choices are
an investment in a forward-looking “location asset” that, in part, reflects expected payoffs coming from
future local job opportunities. Consistent with our analysis of the effect of “career values,” they find that
these forward-looking location characteristics can affect current household choices. We contribute to this
literature by documenting local changes in expected career dynamics over time and how they are affected
by automation shocks, all focusing on a forward-looking measure of individuals’ lifetime income.
Fifth, we also contribute to the literature on populism. Populist movements around the world have
vastly different agendas (Rodrik, 2018), but common identifiers (Guriev and Papaioannou, 2022). Populist rhetoric relies on policies tailored to demote redistributive measures (Drake, 1982; Dornbusch and
Edwards, 1991), exploiting people’s fear of the rich capturing the political process (Dornbusch and Edwards, 1991; Acemoglu et al., 2013; Di Tella and Rotemberg, 2018), economic insecurity (Autor et al.,
2016; Guiso et al., 2017; Panunzi et al., 2020; Dippel et al., 2015), and low trust (Algan et al., 2017). In
line with this literature, the Brexit campaign in the U.K., the anti-immigrant movements in Germany and
France, Erdogan in Turkey, Putin in Russia, and Trump’s anti-trade and anti-immigrant narratives ring
an anti-establishment and anti-elite tone. A growing group of papers estimates the impact of economic
5
shocks related to technology and globalization on the political fortunes of politicians in the US (Autor
et al., 2020a, 2016; Che et al., 2022; Heins, 2023; Frey et al., 2018), Europe (Colantone and Stanig, 2018b;
Anelli et al., 2021), UK (Gallego et al., 2022), Germany (Dippel et al., 2015), and France (Malgouyres,
2017). Enke (2020) argues that increased polarization in moral values contribute to the recent changes in
vote patterns in the United States. We contribute to this literature by studying the impact of forwardlooking economic expectations on electoral preferences for extreme candidates in the US and associated
demand- and supply-side political changes.
The rest of the paper is organized as follows. Section 2 describes the data on job employment histories
and summarizes the basic patterns in job transitions over time. Section 3 describes the theoretical
framework, defines career values, and introduces local labor market career values. Section 4 presents
the empirical strategy that allows us to study the effects of automation at the local level, and provides
descriptive evidence. Here, we also summarize the basic relationship between local labor market career
values and automation. Section 5 studies the mechanisms, and section 6 studies the implications of
changes in career values for long-term decisions. Section 7 discusses the political effects of automation
and the associated changes in career values. Section 8 concludes.
2 Job Transitions Data
Employment History Data We use data from Burning Glass Technologies, one of the largest repositories of resumes available online in the United States. The data set combines resumes of over 16 million
individuals with location and career history, collected from nearly 3,000 job boards, such as monster.com,
LinkedIn, Yahoocareers, and similar sites. The data contain information about the job titles candidates
have (mapped into the standard occupational classification codes, or SOCs) for each job an individual
holds. The data also list the companies that candidates worked at, the job start and end dates, their
education and certification, and their location at a zip-code level, and are parsed to create a personal
career history, starting from the first reported job to the most recent occupation that they hold. For the
years 2000-2016, we can observe individuals coming from 379 out of 388 Metropolitan Statistical Areas
in the United States. Table 1 provides the summary statistics of the data.
Working with a large and detailed resume data set provides us with a number of advantages, most
importantly, being able to see the movement of an individual across occupations and industries. There are
also some concerns related to working with this data set. First, some, such as blue-collar jobs, may not
hire based on individual resumes, and the data set may overrepresent the jobs from some industries and
underrepresent those from others. Figure A.1 in the online appendix compares the share of resumes in the
BGT data relative to the shares reported by the Bureau of Labor Statistics. Relative to the BLS shares, we
see that the data oversamples from white-collar occupations including finance/marketing, IT/engineering,
and management. Schubert et al. (2021) also shows that these occupations are substantially overweighted
in the BGT data. Data underrepresent construction/transportation and hospitality/tourism sectors,
where most occupations can be seasonal. Occupations in other industries seem to have shares comparable
6
Table 1: Summary of the Job Characteristics
Mean SD Min Max N
Female 0.50 0.50 0.00 1.00 14,442,178
Age 28.42 7.00 16.00 94.00 16,737,721
College Degree 0.17 0.37 0.00 1.00 16,737,721
First Year Data 2001.62 7.75 1990.00 2017.00 16,737,721
Last Year Data 2013.46 5.06 1990.00 2017.00 16,737,721
Years of Work 12.84 7.93 1.00 28.00 16,737,721
Average Number of Occupation Changes 2.31 2.63 0.00 92.24 16,737,721
Average Number of Occupation Changes per Year 0.22 0.22 0.00 8.39 16,737,721
Average Number of Moves Up (wage-wise) 0.96 1.21 0.00 40.13 16,737,721
Average Number of Moves Down (wage-wise) 0.74 1.06 0.00 55.61 16,737,721
Average Number of Same Pay Moves 8.04 6.60 0.00 122.00 16,737,721
Number of Unique Occupations (per worker) 2.89 1.57 1.00 17.00 16,737,721
Number of Unique Firms (per worker) 2.91 1.92 0.00 22.00 16,737,721
Total
Number of Unique Workers 16,737,721
Number of Unique Occupations 1,070
Number of Unique Firms 18,100,000
Notes: Statistics are calculated using the resume data from BGT, calculated over the years 2000–2016.
to those of the BLS data.
Second, occupations that require resumes from applicants may be more concentrated in urban areas.
Figure A.2 in the online appendix demonstrates the distribution of the resume sample across counties.
The figure demonstrates that the coverage is better in some locations than others, but the majority of
the highly populated areas of the U.S. are represented in the data.
A third concern with the data is the possibility of false reporting of career history, in particular in
an effort to hide damaging records. Most typically, individuals may not truthfully report job start and
end dates in an effort to hide a gap in employment. For our study, exact dates of job start and end are
not essential, but truthful reporting of the jobs held matters. Since occupational history is often verified
before employment, we believe false reporting of employment history is less likely to be an issue. While
empirical evidence on resume accuracy is scarce, the limited studies that we are aware of do not show
high rates of resume fraud.2 Moreover, even in the case of misrepresentations, the fact that workers claim
a job to be consistent with their work history still constitutes evidence that the stated career transitions
are plausible, which is what we are trying to measure.
Fourth, there may be concerns about the representativeness of the BGT resumes for different demographic groups of workers. Schubert et al. (2021) show that the resume data set is close to being
representative of the overall labor force in terms of gender balance. Moreover, while the raw data overweight younger workers, we explicitly correct for this by computing occupational mobility only after
reweighting observations to adjust for the relative prevalence of different ages in our sample relative to
the U.S. labor force. The BGT resumes also overrepresent workers with a college degree relative to the
2For example, Nosnik et al. (2010) found that only 7% of the publications listed by a sample of urology residency
applicants on their resumes could not be verified – and truly fraudulent listings have to be a subset of these. For a more
detailed discussion of this and other concerns regarding the BGT data, see the online appendix to Schubert et al. (2021).
7
population.
These and other potential demographic imbalances do not represent a serious issue for our analysis,
as we use the data mainly to compute occupational mobility, not to construct representative composition
statistics at the labor market level. While the overall sampling of the BGT resumes may overweight some
occupations (e.g., those with higher education requirements) relative to others, we only require the data
to be representative of worker mobility conditional on starting in each occupation. Moreover, the fact
that the BGT data set has more than an order of magnitude more data points compared to other data
sources, such as the Current Population Survey, and does not rely on the consistent coding of occupations
by surveyors means that our occupation-level transition measures are likely to have substantially lower
measurement error.
Preliminary observations In this subsection, we present some graphical analysis with a description
of the transitions observed in the labor market over the course of the 15 years.3 As Figure 1 suggests,
occupational transitions seem to have slowed during 2010-2015 relative to the 2000-2005 period.4 Such
transitions may be upward—moving toward a better-paying or higher-prestige position—or downward.
Figure 3 illustrates transitions from the “production” occupation group (in the center)—the group to
which most manufacturing occupations belong—to any other occupation group, separately for the years
2004 and 2016 on the left and on the right, respectively. Each circle connected with an arrow indicates the
occupation group workers transition to, and the size of the circle indicates the probability of transitioning
to the group. Comparing the left and right figures, the probability of transitions from production to
“management” and “architecture and engineering” occupation groups—which likely represent upward
transitions—seem to have declined between 2004 and 2016. Appendix figures A.3 and A.4 repeat this
exercise for the construction and installation groups, respectively, and show a similar pattern of reduced
mobility towards management and architecture and engineering jobs.
Figure 4 complements the earlier figures by depicting the trends for transitioning out of production
occupations and transitions to management occupations. This figure shows the numbers normalized
by the total number of transitions, to correct for natural variation coming from the business cycles.
As one can see, the first type of transitions seems to be going up over time, while the second type of
transitions seems to be going down. The normalized transition probabilities for moving to occupations
in management, engineering, construction, maintenance, and production for each 5-year period between
2000 and 2015 are summarized in Figure 5. This figure is, again, consistent with the decline of upward
mobility over time: the transitions to “management” and “architecture and engineering” occupations
seem to be declining over time, while other occupation groups do not demonstrate a clear trend.
As we briefly discuss in the introduction, we can also distinguish between the upward and downward
transitions based on information about the average wage of an occupation. In Figure 2, we summarize
changes as transitions to better-paid jobs, worse-paid jobs, and to (approximately) equal-payment jobs
3For the empirical method details, see Section 3. The trends presented are based on the BGT data.
4The latter period may also include the long-term consequences of the 2008 economic recession. However, since the
recession resulted in job losses, transitions may be expected to intensify in this period.
8
Transitions from Production, 2004 vs 2016 Figure 3: Probabilities of Transitioning out of Production Occupations, 2004 vs 2016
Transitions to Management
Transitions from Construction and Extraction
Transitions from Installation, Maintenance, and Repair
Petrova et al. (2020) Automation, Career Values, and Politics 26/10/2021 9 / 59
(a) 2004 (b) 2016
Notes. Figures show the normalized probabilities of transitioning from production occupations to various other occupation groups. Panel (a) shows the transitions for
the year 2004 and panel (b) shows the transitions for 2016. The size of each bubble indicates the likelihood of transition to the labeled occupation group. Annual job
transition probability out of occupation o is calculated as the share of individuals in the BGT data employed in occupation o in period t and in any occupation other
than o in period t + 1. Transition to occupation o is calculated by taking all the individuals changing occupations between time period t and t + 1 in the BGT data, and
calculating the share employed in occupation o in period t + 1. Normalization is carried out by dividing the transition probabilities between specific occupations by the
overall probability of transitioning between any two occupations between period t and t + 1. Occupations held for less than 180 days are excluded. Plotted probabilities
are moving averages over 3 years around the year of interest.
9
Figure 4: Transitions from Production Occupations and Transitions into Management Occupations
(a) Move from production occupations (b) Move to management occupations
Notes. Panel (a) shows the probability of transitioning from the “production” two-digit occupation group to any other
occupation, and panel (b) shows the probability of moving from any occupation to the management occupations. Annual
job transition probability out of occupation o is calculated as the share of individuals in the BGT data employed in occupation
o in period t and in any other occupation than o in period t + 1. Transition to occupation o is calculated by taking all the
individuals changing occupations between time period t and t + 1 in the BGT data, and calculating the share employed in
occupation o in period t+1. Normalization is carried out by dividing the transition probabilities between specific occupations
by the overall probability of transitioning between any two occupations between periods t and t + 1. Occupations held for
less than 180 days are excluded. Plotted probabilities are moving averages over 3 years around the year of interest.
over 5-year time periods. As one can see, transitions to worse-paid jobs increased, while transitions to
better-paid jobs decreased between 2000-2005 and 2010-2015.5 Overall, the basic descriptive analysis of
occupational transitions presented in this subsection suggests a decline in upward transitions between
2000 and 2015.
Comparison to CPS mobility patterns. While the pattern of declining mobility and increasing
likelihood of occupational transitions to lower-wage jobs is quite striking in our main data set (Figures
1 and 2), there may be a concern that this pattern is a function of our data source. To confirm that
this pattern is robust, we compute similar measures of mobility from Current Population Survey (CPS)
data, which is a much smaller sample than our resume data (and therefore less suitable for the detailed
geographic and demographic breakdowns we are interested in), but is designed to be nationally representative and is commonly used for studying short-run occupational mobility in the aggregate (e.g.,
Moscarini and Vella, 2008). While occupational mobility measurement in the CPS has its own issues related to spurious job changes as a result of measurement error and imputation in the occupational coding
(Moscarini and Thomsson, 2007; Kambourov and Manovskii, 2013), we are not interested in matching
the level of mobility but rather the trend over time. We focus on retrospective measures of occupational
mobility at a 1-year horizon in the biannual Employee Tenure and Occupational Mobility Supplement,
which better captures mobility trends than linking individual responses from the monthly survey over
time (Vom Lehn et al., 2022). We limit the sample to male workers, aged 20-64, who are employed at
5A period of unemployment may result in transitions to temporary, low-wage occupations to ‘make ends meet.’ To avoid
temporary positions, we consider occupations held for a period of 6 months at a minimum. However, such temporary jobs
are less likely to be reported on resumes, as applicants see them as irrelevant or even damaging to their careers.
10
Figure 5: Transitions to Various Occupation Groups, by Time Period
Notes. The figure presents the normalized transition probabilities for moving into occupations in management, engineering,
construction, maintenance, and production for 5-yearly periods between 2000 and 2015. Annual job transition probability
out of occupation o is calculated as the share of individuals in the BGT data employed in occupation o in period t and in
any other occupation than o in time period t + 1. Transition to occupation o is calculated by taking all the individuals
changing occupations between time period t and t+1 in the BGT data, and calculating the share employed in occupation o in
period t+ 1. Normalization is carried out by dividing the transition probabilities between specific occupations by the overall
probability of transitioning between any two occupations between period t and t + 1 (see p. 14 for details). Occupations
held for less than 180 days are excluded. Plotted probabilities are moving averages over 3 years around the year of interest.
the time of the survey and use harmonized 1990 occupation codes provided by IPUMS-CPS. The trends
over time in 1-year occupational mobility out of detailed occupations, as well as out of aggregated broad
occupation groups, in the CPS are shown in Appendix Figure A.6. We find that, similar to the mobility
in the resume data, there is a clear downward trend since the mid-1990s that only partially reverts in
the 2010s. Appendix Figure A.5 shows the change between the early 2000s and early 2010s in mobility
averaged by broad occupation groups, which shows a similar pattern to that seen in Figure 1 and shows
that the finding of declining mobility is not driven only by particular occupations, or by idiosyncrasies
of the resume data. This aligns with other recent papers that find declining occupational mobility since
the 1990s in CPS data (Moscarini and Thomsson, 2007; Xu, 2018; Vom Lehn et al., 2022).6
To compute the CPS analogue of Figure 2, we need occupation information and wage data which
is not contained in the retrospective survey supplements: instead, we use the basic monthly survey for
March of each year and IPUMS provided links between individual respondents over time. With the same
demographic sample restrictions as before, we record the differences in hourly wages between surveys
one year apart, conditional on the recorded harmonized occupation code being different. While the
occupational moves are more likely to contain measurement error in this data, it has the advantage
that we can construct our mobility estimates for a longer time period. Appendix Figure A.7 shows the
likelihood of transitioning into a job with higher, lower, or similar wage to the previous job, conditional
on moving between detailed occupations, for 1981-2020. We find that there has been a marked increase in
the likelihood of experiencing declining wages when switching occupations in the 2000s and 2010s relative
6This may have been preceded by increases in occupational mobility from the 1970s to the 1990s (Kambourov and
Manovskii, 2009).
11
to previous decades in the CPS, and also a decline in the chance of higher wages throughout the 2000s,
which only partially reversed during the 2010s. Overall, these results confirm that the mobility trends in
the resume data are qualitatively very similar to those in the nationally representative CPS data.
However, looking at the probability of one-time transitions does not take into account the lifetime
gains from this transition. For instance, individuals may choose to move to a worse-paying occupation
if it offers better opportunities for upward mobility later. To understand the expected lifetime gains
associated with choosing each occupation, and to compress all the multidimensional information from
the individual job histories, in the following section, we will formally define the concept of career values.
Following that, we will describe the trends in career values in the US and demonstrate a recent decline,
partly stemming from reduced occupational mobility. We will then discuss automation and robotization
as one factor that contributed to this decline.
3 Methodology
3.1 Combining Transitions and Wages: Identifying Career Values
To talk about an individual’s sequence of upward and downward transitions over the lifetime, ideally, we
want to have a single first-approximation index of an individual’s lifetime income given the job that they
currently have. We introduce a measure of career value at the worker-occupation level. Specifically, for
a given worker, we can think of the present monetary value of their lifetime income Cvt in time period t
as the sum of current and future wages W along their career path v, discounted by a factor β:
Cvt =
X
T
j=1
β
j−1Wv,t−1+j . (1)
Neither the worker nor the econometrician can observe the future career value of an individual perfectly. Instead, for an individual in occupation o in period t, we can estimate the present value of her
career by considering the probabilistic path of occupations she is likely to hold in periods t+1, t+2, ... and
the wages these future occupations pay. Specifically, we define expected career value Cot for each occupation o by summing over the average current probabilities and (expected) wages of different occupational
sequences following o, for all occupations {o, p, q, r, . . . NOcc}:
Cot = Wot + β
X
p∈NOcc
π
o→p
t

Wpt + β
X
q∈NOcc
π
p→q
t

Wqt + β
X
r∈NOcc
π
q→r
t
(Wrt + . . .)



 (2)
= Wot + β
X
p∈NOcc
π
o→p
t Cpt, (3)
where π
o→p
t denotes the probability of transitioning from occupation o to p, π
p→q
t denotes the probability
of transitioning from occupation p to q, and so on. Wot is the wage in occupation o in time period t and
β is the discount rate. The right-hand side includes the occupation that the worker is currently in, as
there is some probability π
o→o
that she does not switch occupations.
12
Expected Cot values can be calculated as a function of wages and probabilities by assuming that a
worker can reason through higher-order transitions between occupations. Let Ct represent the present
value of careers, Wt represent the vector of wages, and Πt represent the probability transition matrix
for all occupations in time period t. Then we can rewrite Equation (3) by stacking occupations as:
Ct = Wt + βΠtCt
= Wt + βΠtWt + β
2Πt
2Wt + β
3Πt
3Wt . . .
= (I − βΠt)
−1Wt
= ΨtWt,
where the diagonal of the matrix Ψt captures the importance of the current occupation for the value of a
worker’s future career path and off-diagonal elements represent the weight attached to other occupations
that the worker might move into. Intuitively, for each occupation, the diagonal values subtracted from
one would indicate the probability of moving to other occupations.
We can empirically compute Ct from data on wages and occupational transition probabilities under some assumptions. We compute the transition probabilities using BGT data and contemporaneous
transitions as a proxy for expected future transitions. For an average worker, the best predictor of the
distribution of potential occupational transitions in the future is likely the current observed distribution
of occupational transitions. For the expected wages Wt after transitioning into another occupation, we
use the most recent wages as a proxy.
The assumption that future transition probabilities and wages can be proxied by today’s values can
be justified if workers calculate career expectations based on their recent observations, rather than using
a sophisticated model to project the evolution of the US labor market. In other words, we assume that
in the absence of perfect foresight into future labor market conditions, workers use their most up-todate information to construct their beliefs about future earnings.7 8 We further assume that workers
can anticipate the transitions that follow an immediate transition. This is a realistic assumption since
Table 1 indicates the average occupational transitions per worker is 2.89.
To compute occupational transitions, we approximate the share of workers moving from occupation
o to occupation p with the share of all workers observed in occupation o at any point in year t who
are observed in occupation p at any point in year t + 1, dropping jobs lasting less than 6 months. We
only focus on transitions between occupations from one year to another and exclude any other possible
transitions, including those into/from unemployment, which are usually not recorded on resumes.9 This
7Experts with advanced degrees often do not agree about the growth trends in the labor markets (e.g., Graetz, 2020),
and we presume that the workers also lack perfect foresight about it.
8We also assume that workers use the wages in their local market to calculate future career expectations rather than the
wages in other local markets. This implicitly assumes that workers consider geographic moves too unlikely to be relevant in
this calculation.
9Note that as long as expectations of transitions into unemployment do not vary substantially between occupations, they
do not affect the relative career values of different occupations.
13
unconditional occupational transition probability is our proxy for πo→p:
π
o→p
t =
Number of workers in occ o in year t who are observed in occ p in year t + 1
Number of workers observed in year t + 1 who were in occ o in period t
.
We use full-time jobs and their 6-digit Standard Occupational Classification (SOC) codes defined by the
BLS. To reduce noise in our estimates, we average wages and transition probabilities over the adjacent
years before and after t, centering around the year of interest. For instance, our estimate of 2004 transition
probabilities uses data from 2003–2005 period.10
The above transition probabilities treat all local labor markets the same in terms of how likely it is for a
worker to obtain a job in another occupation. This is unrealistic, because local careers are constrained by
the occupations that exist in local industries, and, as geographic mobility tends to be small, occupations
in other locations are likely of little relevance for most workers. To account for the local market variation,
we adjust the occupational transition probabilities by the relative prevalence of occupations in the local
labor market. This reflects the intuition that in a location where another occupation is twice as prevalent
as the national average, that occupation should also be twice as likely to be the next career step—relative
to the average national tendency to move into that occupation. Local career transition probabilities in
market c are captured by the matrix Πct, where each element (o, p) of Πct is defined as
π
o→p
ct =
sp,ct
sp,t
π
o→p
t
P
j∈NOcc
sj,ct
sj,t
π
o→j
t
.
Here, sp,ct represents the local and sp,t represents the national employment share of occupation p, so that
each element of Πct scales the corresponding element of Πt by the local prevalence and then renormalizes
each row to sum up to one, considering all occupations j.
To adjust the transition probabilities for the local prevalence of occupations, we use ACS microdata
on occupation shares in each CZ relative to the national employment share, and fill in occupations not
contained in the ACS data by computing the analogous ratio of the occupation share for the majority
state of the CZ from BLS Occupational Employment Statistics (OES) data, again relative to the national
average. If data from neither source is available, we assume that the occupation does not exist locally
and set its share to zero.
We combine the transition probabilities with annual wages by occupation and state, using data provided by the BLS Occupational Employment Statistics survey, thus we use localized (not national-level)
wages. We adjust all wages for inflation according to the urban consumer price index (CPI) and we set
the discount factor to β = 0.85.
11
10Our methodology for estimating the job transitions from BGT data is similar to Schubert et al. (2021), Section 2.
11This value is consistent with experimental estimates of discount factors applied in economic decisions (Coller and
Williams, 1999; Newell and Siikamäki, 2015; Patnaik et al., 2022). It is on the lower end of values exogenously imposed in
discrete choice models in the literature (e.g. in Dix-Carneiro (2014)), as we wanted to be conservative in the degree to which
we assume that future career moves matter to workers.
14
3.2 Career Values, Wages, and Automatability
Figure 6 shows the average change in career values between 2000 and 2016 vs. the average starting
annual wage between 2000 and 2002 for each occupation group. This figure shows that career value
changes are correlated with the initial wages, but there is substantial variance: for some occupations,
career values increased substantially, while for others they stayed the same or even declined. In contrast,
if we plot changes in wages versus initial wages, the dots in the graph seem to be well aligned along a
linear prediction (Figure 7). The comparison of Figures 6 and 7 suggests that occupations are different
not only in the wage gains, but also in the career mobility opportunities they offer, and the career values
of occupations in Figure 6 are more dispersed.
Figure 6: Average Career Value Change Between 2000 and 2016 vs. Starting Wage, by 2-digit SOC
Management
Business & Financial Operations
Computer & Mathematical
Architecture & Engineering
Life, Physical, & Social Science
Community & Social Svc
Legal
Education, Training, & Library
Arts, Design, Entert., Sports, & Media
Healthcare Practitioners and Technical
Healthcare Support
Protective Service
Food Prep. & Serving
Building & Maint.
Personal Care & Service
Sales
Office & Admin. Support
Farming, Fishing, & Forestry
Construction & Extraction
Installation & Repair
Production
Transport. & Material Moving
-6000
-4000
-2000
0
2000
4000
6000
8000
10000
12000
14000
Avg. Chg. in Career Value '01-'16 (USD, CZ avg.)
10000 20000 30000 40000 50000 60000 70000 80000
Annual Wage (USD, 2001)
Notes. The figure depicts the average change in career values between 2000 and 2016 vs. the starting wage of an occupation,
where the latter is calculated by averaging the wages in an occupational group for the years 2000 to 2002. Career values are
calculated using the transition probabilities from BGT data and annual wage data from the BLS Occupational Employment
Statistics survey. All wages are adjusted for inflation according to the urban consumer price index (CPI). The discount
factor in the calculation of the career values is set to β = 0.85.
In appendix Figure A.8, we show that the change in occupational career values between 2000 and
2016 is correlated with the automatability of an occupation, a computerization measure introduced by
Frey and Osborne (2017) using a combination of expert opinions and machine learning. We find a clear
negative relationship between automatability and the career value of the occupation groups. According
to the figure, an approximate 0.2 unit increase in the automatability score is associated with $2,500 less
15
Figure 7: Average Wage Change Between 2000 and 2016 vs Starting Wage, by 2-digit SOC
Management
Business & Financial Operations
Computer & Mathematical
Architecture & Engineering
Life, Physical, & Social Science
Community & Social Svc
Legal
Education, Training, & Library
Arts, Design, Entert., Sports, & Media
Healthcare Practitioners and Technical
Healthcare Support
Protective Service
Food Prep.
Building & Maint.
Personal Care
Sales
Office & Admin. Support
Farming, Fishing, & Forestry
Construction & Extraction
Installation & Repair
Production
Transport
-1000
1000
3000
5000
7000
9000
11000
13000
Avg. Chg. in Annual Wage '01-'16 (USD, CZ avg.)
15000 25000 35000 45000 55000 65000 75000
Annual Wage (USD, 2001)
Notes. Figure plots the average change in wages over the period 2000 to 2016 vs. the starting wage calculated by averaging
the wages in the occupational group for the years 2000 to 2002. Annual wage data is gathered from the BLS Occupational
Employment Statistics survey. All wages are adjusted for inflation based on the urban consumer price index (CPI).
in upward career value change between 2000 and 2016.
Local Market Career Values To study the changes in career values at the local labor market level,
we estimate the “local market career value” (LMCVct) which represents the aggregate career values of
all local occupations weighted by their local employment shares in CZ c in time period t:
LMCVct =
N
XOcc
o∈NOcc
λcotCcot, (4)
where λcot is the share of employees holding occupation o in market c in period t, Ccot is the career value
of occupation o in period t in local market c. Here, LMCVct has a straightforward interpretation: it
measures the present discounted value of expected lifetime earnings for a randomly selected worker in
labor market c. We focus on CZs as the unit of local labor markets in line with Autor et al. (2013)
and Acemoglu and Restrepo (2020). For employment shares in CZs, we rely on data from the American
Community Survey (ACS).
The change in LMCVct can be decomposed into changes due to the composition of local labor market
16
employment opportunities and career value changes:
∆LMCVct =
N
XOcc
p∈NOcc
∆λcotCco,t−1
| {z }
Market composition change
+
N
XOcc
p∈NOcc
λco,t−1∆Ccot
| {z }
Career value change
+
N
XOcc
p∈NOcc
∆λcot∆Ccot,
| {z }
Interaction: Market comp. × CV
where the first term “market composition” refers to the changes in a local market due to the shifts in the
employment shares of occupations (∆λcot), holding the probability of transitions between occupations and
the wages fixed. The local market career value could increase, for instance, due to composition changes
if the shares of high-value occupations increase in a region. The second term “career value” captures the
changes in a region due to the shift in career values, fixing the job composition. This term describes
the changes that occur either due to shifts in the probabilities of transitioning between occupations or
due to changes in the anticipated wages of future occupations while holding the current distribution of
occupations in the local labor force constant. The last interaction term encompasses changes that occur
in tandem due to the interaction between shifts in market composition and changes in career value.
The changes in career value can further be decomposed into changes in occupation o in location c due
to changes in career path mobility and changes in wages, as follows:
∆Ccot =
X
O
∆ψ
o→p
t Wcp,t−1
| {z }
Career path chg.
+
X
O
ψ
o→p
t−1 ∆Wcp,t
| {z }
Wage changes
+
X
Occ
∆ψ
o→p
t ∆Wcp,t
| {z }
Interaction: Wage × Path
where the ψ
o→p
t
represents the career mobility weights of the occupation p that individuals transition to
from occupation o, i.e. the (o, p) element of the matrix Ψt = (I − βΠt)
−1
.
Table 2 summarizes the changes in career values and their components over the years, averaging across
all CZs. The results imply that career values were growing in the 2000-2008 period, while there is some
decline in the 2008-2016 period. Decomposition into wage and occupational path changes indicates that
wages continued to grow in both periods, albeit at a lower rate in the later period. However, there was a
loss in career values from the occupational mobility component in both periods, which implies a decline
in transitions into higher-paying occupations.
According to these numbers, between 2000 and 2008, overall, the average LMCV increased by $2,100;
whereas from 2008 to 2016, it declined by $200. The average local market career values were $230,957
in 2000, $233,025 in 2008, and $232,874 in 2016. Thus, from 2000 to 2008, there was an approximate
0.9% increase in lifetime career value, and from 2008 to 2016, there was an approximate 0.09% decline,
on average. The direct effect of wages on the average LMCV was a $16,100 (7%) increase between 2000
and 2016. The direct effect of occupational mobility on the average LMCV was a $12,500 (5.5%) decline
in the same period. In the discussion which follows, we will look at the effects of robotization on career
value declines, and show that the effects of the latter are more dramatic for regions undergoing a higher
rate of robotization.
17
Table 2: Average Local Market Career Value Changes: Decomposition
2000-2008 2008-2016 2000-2016
Local Mkt. CV Change 2.1 -0.2 1.9
Mkt Composition 0.4 1.4 1.0
Occ. Career Values 2.8 -0.7 1.4
Career Path chg. -5.8 -7.0 -12.5
Wage chg. 9.0 6.9 16.1
Own occ. wage chg. 1.8 1.8 3.5
Other occ. wage chg. 7.2 5.1 12.6
Interaction: Path x Wages -0.4 -0.5 -2.3
Interaction: Comp. x Occ. CV -1.1 -0.9 -0.4
Notes: All terms are in units of $1,000 of net present career value, at a discount factor of 0.85, and assuming rational
expectations of career values.
3.3 Identifying Effects of Robot Adoption
Throughout the paper, we aim to study the impact of robotization, an important factor of economic
transformation that has been taking place during our time period (2000-2016). In most specifications,
we follow Acemoglu and Restrepo (2020) and Faber et al. (2019) and focus on a measure of exposure in
CZs, weighted by an earlier period industry shares for each industry potentially affected by robotization.
CZs differ in their exposure to automation because they are composed of different industries, and workers
across CZs are exposed to robot adoption at different rates. Our empirical strategy consists of estimating
regression specifications of the form:
∆Outcomec,(t0,t1) = β1 + β2Exposure_to_robotsc,(t0,t1) + βXc + ε
`
c,(t0,t1)
, (5)
where ∆Outcomec,(t0,t1)
stands for the aggregate change in labor market outcomes of interest between
periods t0 and t1 in CZ c, Xc stands for controls for c for period t1 and includes the log of population,
share of males (in total population), share of population above 65 years, share of population with high
school, some college, college and postgraduate education, share of whites, African Americans, Hispanics,
and Asians (in total population), share of employment in manufacturing, construction, and mining, and
share of female workers in manufacturing employment, following the list of controls used in Acemoglu
and Restrepo (2020) and Faber et al. (2019). In addition, for some specifications, we include share of
employment in manufacturing, share of female employment in manufacturing, share in light manufacturing (textile industry and the paper, publishing, and printing industry), share of employment in routine
occupations and change exposure to imports from China from 1990 to 2007 using data from the American
Community Survey and County Business Patterns.
For the industrial robots in stock, we use data provided by the International Federation of Robotics
(IFR). The dataset covers 38 industry codes with the International Standard Industrial Classification
(ISIC) code. While the data contain information about the operational stock of industrial robots in
about 50 countries between 1993 and 2016 (corresponding to about 90% of the industrial robots market),
18
the industry breakdown of the robot stocks starts in 2004. Therefore, we focus on the data after 2004 to
calculate automation exposure in the US For more details on this dataset, please see appendix B.12
Exposure to robots in CZ c in period (t0, t1), Exposure_to_robotsc,(t0,t1)
, is constructed by weighting
over local industry employment shares and the change in the robot stock impacting a particular industry
code as in Acemoglu and Restrepo (2020) and Faber et al. (2019).:
Exposure_to_robotsc,(t0,t1) =
X
i∈ι
lciAP Ri,(t0,t1),
(6)
where AP Ri,(t0,t1)
is the change in the adjusted penetration of robots in an industry i between t0 and t1.
Because adoption of robots in any industry is likely to be jointly determined with labor market outcomes
through a host of correlated unobservables—that is, Cov(ε
`
c,(t0,t1)
, Exposure _to_robotsc,(t0,t1)
) 6= 0—,
we might not be able to isolate the changes in local labor market conditions due to automation shocks
if the above-mentioned robot exposure measures are calculated using data from the US13 To alleviate
this problem, we use a Lasso-based IV approach (Chernozhukov et al., 2015), which chooses the set of
countries C for which the adjusted penetration of robots linearly best predicts the US robot adoption
measure AP Ri,(t0,t1)
.
The identification of causal effects of automation exposure on labor market outcomes requires that
(1) the instrument based on exposure to European robot adoption predicts exposure to the US variation
in robot adoption, and (2) the local exposure to industries with growing robot adoption in Europe is
uncorrelated with unobservable shocks driving local labor market dynamics in the US. That is, intuitively,
we can think of the IV Lasso procedure as constructing a first-stage predicted variation in local robot
exposure given by
Exposure_to_robotsc,(t0,t1) =
X
j∈|C|
β
Lasso
j
 X
i∈ι
lciAP Rj
i,(t0,t1)
!
, (7)
where |C| is the set of European countries selected by the first stage of an IV Lasso procedure, and β
Lasso
j
are the corresponding weights on the exposure to each country’s robot adoption. The exclusion restriction
12We handle the limitations with the IFR data similarly to Acemoglu and Restrepo (2019) and collapse industries into the
following categories: 1) agriculture, forestry, fishing; 2) mining and quarrying; 3) food and beverages; 4) textiles; 5) wood
and furniture; 6) paper; 7) plastic and chemical products; 8) glass, ceramics, stone, mineral products (non-auto); 9) basic
metals; 10) metal products; 11) industrial machinery; 12) electrical/electronics; 13) automotive; 14) other vehicles; 15) all
other manufacturing branches; 16) electricity, gas, water supply; 17) construction; 18) education/research/development; 19)
all other non-manufacturing branches. Acemoglu and Restrepo (2019) note that about 30% of robots are not assigned to
an industry. Similar to Acemoglu and Restrepo (2020), we assign unclassified robots to industries in the same proportion
as the classified data. Also, similar to the authors, we use the overall stock of robots for North America to measure the US
exposure to robots. The authors state that this is not a concern “since the United States accounts for more than 90% of the
North American market.”
13To deal with this issue, Acemoglu and Restrepo (2020) use an average penetration measure from a subset of European
countries which are well ahead of the United States in their adoption of industrial robots. Using robot penetration of
European countries thus allows to “isolate the source of variation coming from global technological advances (rather than
idiosyncratic US factors)” (pg.13 Acemoglu and Restrepo, 2020). They calculate the adjusted penetration value using data
from Denmark, Finland, France, Italy, and Sweden. For the time period in our analysis, the same five countries are no longer
necessarily ahead of the United States in automation. As a result, their adoption patterns might become less predictive of
which industries in the US should see an increase in robot adoption.
19
requires Cov(ε
`
c,(t0,t1)
, Exposure_to_robotsc,(t0,t1)
) = 0, conditional on the included control variables.
This approach is similar to other shift-share approaches in the literature where the identification requires
exogeneity of the exposure shares—here, the local industry structure—with regard to the changes in
unobserved local shocks. The econometrics of this approach are detailed in Goldsmith-Pinkham et al.
(2020).
How plausible is the assumption of exogenous industry shares? Acemoglu and Restrepo (2020) already
address a number of potential concerns, such as possible pre-trends in regions more exposed to industries
that are adopting robots at a faster pace. In addition, Goldsmith-Pinkham et al. (2020) suggest that
controlling for coarser industry structures (e.g., manufacturing shares) may help to address confounders
related to overall economic structure. We address this issue by splitting the sample into high and low
manufacturing share areas in subgroup analyses to establish that the effects hold even within the sample
of high manufacturing areas that would be expected to bear the brunt of automation effects. In addition,
our baseline analyses include a rich set of demographic control variables and census division-level fixed
effects that capture differences between the broad US regions (e.g., “Rust Belt” vs “New England”).
3.4 Other Data
In addition to BGT transitions data, wage data from the BLS Occupational Employment Statistics, and
robotics data, we also employ a number of reasonably standard datasets to construct control variables
and additional economic outcomes.
Current Population Survey We use data from the Current Population Survey (CPS) to compute
estimates of occupational mobility that can be compared with estimates from the BGT data. We use
individual-level data on current and previous year occupations from the Employee Tenure and Occupational Mobility Supplement of the CPS for the years 1995-2022, available via IPUMS. The sample is
composed of employed male workers, aged 20-64, who are employed at the time of the survey. The data
uses 2000 SOC occupation codes, which we crosswalk to their 1990 counterparts. We also include data
from the basic monthly survey for March of each year to study changes in wages by occupation.
Bureau of Labor Statistics Occupational Employment and Wage Statistics The Bureau of Labor Statistics (BLS) Occupational Employment and Wage Statistics (OEWS) data, accessed through the
BLS website, is used to compare the share of employment in different occupations in 2016. Occupations
are defined based on 2-digit SOC codes.
Robot Stock Data. Data on the robot stock and robots in operation are from the International
Federation of Robotics data, reported for various countries annually. For more details on these data,
please see Acemoglu and Restrepo (2020) and online appendix B.
Local business dynamics and demographics. We use the County Business Patterns (CBP) data
from the Census to measure county-level employment, wages, and number of establishments by NAICS
20
industry. The data are at the county-level, and we aggregate values to the 1990 commuting zone level. We
measure local employment shares for 2000 and 2010 by demographic group using American Community
Survey microdata accessed through IPUMS. These data can be accessed via IPUMS.
Spending and investment activity data from NielsenIQ LLC and US Census. We construct
data on consumer behavior in various dimensions: we use NielsenIQ Consumer Panel data to measure
changes in household spending 2004-2016. These data are accessed through the Kilts Center at the
University of Chicago Booth School of Business.
New housing construction permits by commuting zones are constructed from county-level counts of
permits available from the Census Building Permits Survey for 1990-2018, aggregated at the level of 1990
commuting zone. The data are available through the US Census Bureau. Some counties do not report
permit numbers. Thus, in cases where not all component counties of a CZ are reporting, we scale the
reported numbers in proportion to the reporting counties’ share of the CZ population to make up for the
missing share, using county populations in 2000 as weights.
Gallup Poll Social Series (GPSS) We use data from the Gallup Poll Social Series (GPSS) to study
effects on people’s perception of the economy. The data are from 2016 and respondents are asked about
whether they think the economy has improved or worsened. Access to the GPSS is through the Penn
Libraries of the University of Pennsylvania.
Political Outcomes We study political outcomes such as voter shares, voting intentions, political ideologies and campaign visits. Data on electoral outcomes are from Dave Leip’s Atlas of the US Presidential
Elections, referred to as the Election Atlas. We use data from the Cooperative Election Study, which
was previously called the Cooperative Congressional Election Study (CCES), to measure people’s voting
intentions in US presidential elections. Data on campaign visits are from Devine (2018). We measure political ideologies by using NOMINATE and Nokken and Poole scores for the US House of Representatives
for the year 2016 available via VoteView.
4 Empirical Results
The previous section showed that measuring career values and occupational mobility provides a new lens
on recent labor market dynamics. We use this new approach to empirically study the effects of one of
the key technological trends of the recent decades: exposure to automation and robotization.
The analysis estimates the effect of robot exposure on labor market outcomes—consisting both of
changes in contemporaneous outcomes such as unemployment and wages, as well as forward-looking
labor market characteristics—in the form of expected career trajectories and the present value of career
income.
21
Robot Adoption and Local Market Career Values
In what follows, our objective is to understand whether an important contemporary technological process,
automation, and robotization, played a role in the observed decline in career values. We start by reporting
how robotization exposure influences career values in a typical commuting zone, estimating equation (5)
with change in local market career values, ∆LMCVc,(t0,t1)
, as an outcome.
Table 3 reports this relationship across different years using OLS (columns 1-3) and IV Lasso (columns
4-6). OLS coefficients are slightly smaller than the IV coefficients, which may point to measurement error,
but we do not observe a large bias for OLS estimates in either direction. Coefficients in columns (4)-(6)
suggest that higher exposure to robots resulted in a decline in the expected LMCVs. The shrinkage in
career value is significant for all three periods (2004 to 2008, 2008 to 2016, and 2004 to 2016), and the
magnitudes imply that a one-unit increase in robotization exposure (1 additional robot per 1000 workers)
decreased the average local market career value by $3.9K between 2004 and 2008 and by $2.48K between
2008 and 2016. These values correspond to about 1.7% and 1.1% of the average career value in 2000
($230,957). Table A.18 in the appendix replicates these results but includes the share of employment in
routine occupations and change exposure to imports from China from 1990 to 2007 as controls.
Table 3: Exposure to Robots and Labor Market Career Value Change (OLS and IV Lasso)
Local Market Career Value (OLS) Local Market Career Value (IV)
∆ LMCV ∆ LMCV ∆ LMCV ∆ LMCV ∆ LMCV ∆ LMCV
(1) (2) (3) (4) (5) (6)
U.S. Robot Exposure ’04-’08 -0.370*** -0.390***
(0.074) (0.082)
U.S. Robot Exposure ’08-’16 -0.191* -0.248***
(0.108) (0.095)
U.S. Robot Exposure ’04-’16 -0.318*** -0.338***
(0.041) (0.037)
Mean of D.V. -0.124 -0.015 -0.139 -0.124 -0.015 -0.139
F-stat. 47.369 4.510 63.210
Census Division FE Yes Yes Yes Yes Yes Yes
A&R Controls Yes Yes Yes Yes Yes Yes
Observations 722 722 722 722 722 722
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in parentheses
and clustered by state. All columns include census division dummies as well as controls for demographic characteristics
of commuting zones in 1990 identical to those in Acemoglu and Restrepo (2020) (log population; the share of females;
the share of population above 65 years; the shares of population with no college, some college, college and professional
degrees, and masters and doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment
in manufacturing; the share of female employment in manufacturing; and the share of employment in light manufacturing,
namely, textile industry and the paper, publishing, and printing industry). U.S. Robot Exposure is instrumented using robot
exposure in Denmark, Italy, Sweden, Finland, France, Germany, Spain and the UK in the same time periods. Observations
are weighted by commuting zone population.
Changes in LMCV from automation can be due to changes in market composition, changes in occupation career value, or both. As discussed, the former measures the changes in career values due to the
changes in local market job composition, fixing the probabilities of job transitions and wages. The latter
22
measures the changes in the LMCV due to changes in career values, fixing the labor market composition.
We further decompose the change in career value into changes due to ‘career path’ (ψ) and ‘wages’ (W)
and also report the interactions of these effects.
Table 4 summarizes how the results of this decomposition depend on robot exposure, using the IV
Lasso framework. The IV results for all CZs indicate that the LMCV does not change substantially due
to the market composition, with our confidence intervals allowing us to rule out the effect of up to $66.8
for a mean dependent variable of $2,147. This suggests that migration, though important (Faber et al.,
2019), is unlikely to explain our career values results. So the decline in LMCV is almost entirely due
to the changes in occupation-level career value (column 2), with approximately two-thirds of the effect
($2,040 decline) being due to the decline in wages, and one-third of the effect ($1,170 decline) being due
to the decline in career path, i.e., a decline in upward transitions if we hold wages fixed (both numbers
are for 1 additional robot per 1000 workers).
Repeating the analysis separately for the high-manufacturing and low-manufacturing commuting
zones (2nd and 3rd panels), we see that the estimated effects are mostly coming from the former, albeit the implied first stage for the IV Lasso procedure is much weaker for low-manufacturing commuting
zones, which is in line with intuition.
Table A.2 in the online appendix further decomposes the changes in career values into those from an
employee’s current (own) occupation and from other (future) occupations. In regions with higher robot
exposure, both the lifetime earnings from one’s own occupation and other occupations declined more
steeply. However, the point estimate of the $ decline from other occupations (-0.292) from one additional
robot per 1000 workers is about 6 times that for the current occupation (-0.043). The decline in the
current occupations’ contribution to one’s career value is due to the decline in wages (column 3), whereas
the decline in other occupations’ contribution to career value is associated with both the losses in wages
(column 6) and the reduced job mobility (column 7). Put differently, regions with higher robot exposure
experienced a decline in mobility between occupations: the likelihood of staying in the same occupation
is higher, and transferring to another occupation is less likely.
Automation Exposure and Job Transitions
To better understand the results in Table 3 and Table 4, we look into the impact of exposure of robots
on job transitions only, using the same general framework of equation (5). We normalized the number
of job transitions in each category to have a mean of zero and a standard deviation of 1, to make the
numbers comparable across different columns.
In particular, we focus on transitions to better-paid occupations, same-, and lower-wage occupations,
using the same data as in Figure 2. These results are summarized in Table 5. As one can see, transitions to
same-pay (column 1) and lower-pay (column 3) jobs increased and transitions to lower-wage occupations
increased overall (column 2). These effects are in the same direction in high-manufacturing commuting
zones (columns 4, 6, and 8), but there are no significant changes in the low-manufacturing regions
(columns 5, 7, and 9). The magnitudes imply that one additional robot per 1000 workers leads to on
23
Table 4: Exposure to Robots and Disaggregated Changes of Labor Market Career Values, IV Lasso
∆ CC ∆ CV ∆CV : ∆W ∆CV : ∆ψ ∆CV : ∆ψ × ∆W ∆ CC × CV
Mkt. Compos. Occ. Career Values Wage Career Path
(1) (2) (3) (4) (5) (6)
Panel A: All Commuting Zones
U.S. Robot Exposure ’04-’16 0.005 -0.334*** -0.204*** -0.117*** -0.013* -0.008**
(0.004) (0.038) (0.045) (0.022) (0.007) (0.003)
Mean of D.V. 0.215 -0.244 1.072 -1.140 -0.176 -0.110
F-stat. 0.937 76.622 58.281 19.728 7.107 6.796
Observations 722 722 722 722 722 722
Panel B: High-Manufacturing Commuting Zones
U.S. Robot Exposure ’04-’16 0.002 -0.355*** -0.218*** -0.120*** -0.016 -0.006**
(0.007) (0.031) (0.034) (0.024) (0.010) (0.003)
Mean of D.V. 0.228 -0.630 0.666 -1.152 -0.144 -0.084
F-stat. 0.136 67.760 49.422 15.719 4.309 4.060
Observations 359 359 359 359 359 359
Panel C: Low-Manufacturing Commuting Zones
U.S. Robot Exposure ’04-’16 0.244 0.405 0.414 0.074 -0.084 0.346*
(0.288) (1.808) (1.117) (1.197) (0.349) (0.181)
Mean of D.V. 0.199 0.123 1.466 -1.138 -0.206 -0.135
F-stat. 1.524 0.146 0.365 0.010 0.131 13.275
Observations 360 360 360 360 360 360
Census Division FE Yes Yes Yes Yes Yes Yes
A&R Controls Yes Yes Yes Yes Yes Yes
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets
and clustered by state. All columns include census division dummies as well as controls for demographic characteristics
of commuting zones in 1990 identical to those in Acemoglu and Restrepo (2020) (log population; the share of females;
the share of population above 65 years; the shares of population with no college, some college, college and professional
degrees, and masters and doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment
in manufacturing; the share of female employment in manufacturing; and the share of employment in light manufacturing,
namely, textile industry and the paper, publishing, and printing industry). U.S. Robot Exposure is instrumented using robot
exposure in Denmark, Italy, Sweden, Finland, France, Germany, Spain and the UK in the same time periods. Commuting
zones are classified as high manufacturing zones if the share of population employed in the manufacturing sector is greater
than the median share of employment in manufacturing in the country. Low manufacturing zones have below median share
of employment in manufacturing. Observations are weighted by commuting zone population.
average a 7.2% increase in the number of transitions to similar pay jobs (column 1), a 6.2% decrease in
the number of transitions to better-paid jobs (column 2), and a 5.9% increase in the number of transitions
to lower-paying jobs (column 3). For high-manufacturing commuting zones, the corresponding estimates
are a 16.9% increase in the similar pay jobs (column 4), a 6.3% decline in transitions to higher-pay jobs
(column 6), and a 10.2% increase in transitions to lower pay jobs (column 8). Table A.19 in the appendix
replicates these results but includes the share of employment in routine occupations and change exposure
to imports from China from 1990 to 2007 as controls.
Table A.3 in the online appendix provides estimates for the effect of robotization on the probability of
transitioning from manual, cognitive, routine, and non-routine occupations among the individuals transitioning to another occupation, using classifications based on the classification in Autor et al. (2003).
The table demonstrates that in regions with higher exposure to robotization, the likelihood of moving
away from routine and non-routine manual occupations (columns 1, 5, and 6) and manufacturing occupations (column 9) increased significantly, while no significant changes were recorded for the cognitive
24
Table 5: Exposure to Robots and Job Transitions. IV Lasso
Probability of transitions to: Probability of transitions to: Probability of transitions to: Probability of transitions to:
equal wage higher wage lower wage equal wage occupations: higher wage occupations: lower wage occupations:
occupations occupations occupations
high-manuf. low-manuf. high-manuf. low-manuf. high-manuf. low-manuf.
(1) (2) (3) (4) (5) (6) (7) (8) (9)
U.S. Robot Exposure ’04-’16 0.072** -0.062*** 0.059** 0.169*** 1.669 -0.063*** -0.159 0.102*** 0.400
(0.031) (0.010) (0.025) (0.019) (1.055) (0.024) (0.221) (0.018) (0.335)
Mean of D.V. 0.007 0.008 0.008 0.260 0.073 0.263 0.073 0.263 0.074
F-stat. 25.333 86.895 16.673 161.287 32.425 35.045 1.026 42.940 4.847
Census Division FE Yes Yes Yes Yes Yes Yes Yes Yes Yes
A&R Controls Yes Yes Yes Yes Yes Yes Yes Yes Yes
Observations 722 722 722 359 360 359 360 359 360
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets
and clustered by state. All columns include census division dummies as well as controls for demographic characteristics
of commuting zones in 1990 identical to those in Acemoglu and Restrepo (2020) (log population; the share of females;
the share of population above 65 years; the shares of population with no college, some college, college and professional
degrees, and masters and doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment
in manufacturing; the share of female employment in manufacturing; and the share of employment in light manufacturing,
namely, textile industry and the paper, publishing, and printing industry). All columns include controls for the total number
of career values with potential transition to equal, high or lower wage occupations for the years 2004 and 2016 depending on
the outcome variable. In columns (4)-(9), these controls are for the high and low manufacturing zones exclusively. Columns
(4)-(9) also control for the overall probability of transition to equal, high or lower wage occupations in 2004. U.S. Robot
Exposure is instrumented using robot exposure in Denmark, Italy, Sweden, Finland and France in the same time periods.
Commuting zones are classified as high manufacturing zones if the share of population employed in the manufacturing sector
is greater than the median share of employment in manufacturing in the country. Low manufacturing zones have below
median share of employment in manufacturing. Observations are weighted by commuting zone population.
occupations (columns 2, 7, and 8). Table A.4 estimates the effect of exposure to robots separately for
on occupational change (column 1) and employer change (column 2), as well as the changes of employer
within an occupation (column 3) and changes to occupations within one’s current employer (column 4).
We find that higher robotic exposure is associated with a higher likelihood of changing employers and
lower likelihood of changing occupations overall. These results are consistent with our interpretation
that upward job transitions declined as a result of automation and robotization, and this is a reason
contributing to overall decline in upward mobility observed in recent years.
College Premium, Heterogeneity, and Inequality
Our results so far suggest that between 2004 and 2016, the growth in expected lifetime incomes slowed
down, while opportunities for upward mobility steadily declined. Tables 3, 4 and 5 imply that robotization
is one reason for that. In this subsection, we analyze whether education, and more generally other sources
of upward mobility, can mitigate the negative effect of robot exposure.
Career Values by Education and Automation. One of the principal sources of upward mobility
is education. Getting higher education vastly improves workers’ career options, and the college premium
effects are well documented in the literature. We start our analysis by computing career values separately
for different levels of education to see whether exposure to robots affects high- and low-skilled workers
differently. Following the convention in the literature, we define high-skilled workers as those with at
25
least college-level education.14
Table 6 summarizes these results. Column (1) shows the results of the estimation of equation (5)
for the change in occupational composition, column (2) presents the baseline effect for career values,
disaggregated by education, column (3) and (4) show how the components of career values change in
response to robotization, focusing on the changes in wages, holding career transitions constant (column
3) and the changes in career paths, holding the wages constant (column 4). Columns (5) and (6) show
the results for the interactions.
Table 6: Exposure to Robots and Labor Market Career Values, by Education
Occ. Composition Career Value Wage Career Path Career Path x Wage Occ. Comp. x Career Value
∆CC ∆CV ∆CV : ∆W ∆CV : ∆ψ ∆CV : ∆ψ × ∆W ∆CC × ∆CV
(1) (2) (3) (4) (5) (6)
Panel A: Below College
U.S. Robot Exposure ’04-’16 -0.005 -0.388*** -0.237*** -0.152* 0.001 -0.014***
(0.007) (0.095) (0.046) (0.078) (0.013) (0.004)
Mean of D.V. 0.345 -0.162 1.162 -1.120 -0.205 -0.415
F-stat. 0.388 27.243 72.338 4.908 0.550 5.570
Observations 722 722 722 722 722 722
Panel B: College and Higher
U.S. Robot Exposure ’04-’16 -0.008 -0.341*** -0.241*** -0.104*** 0.004 -0.010***
(0.007) (0.032) (0.041) (0.030) (0.006) (0.004)
Mean of D.V. -0.033 -0.189 1.163 -1.188 -0.164 -0.039
F-stat. 0.543 66.896 72.950 13.638 0.134 6.503
Observations 722 722 722 722 722 722
Census Division FE Yes Yes Yes Yes Yes Yes
A&R Controls Yes Yes Yes Yes Yes Yes
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets
and clustered by state. All columns include census division dummies as well as controls for demographic characteristics
of commuting zones in 1990 (log population; the share of females; the share of population above 65 years; the shares of
Whites, Blacks, Hispanics, and Asians; the share of employment in manufacturing; the share of female employment in
manufacturing; and the share of employment in light manufacturing, namely, textile industry and the paper, publishing, and
printing industry). The set of controls is more parsimonious to avoid taking education split and its correlates to take into
account twice. U.S. Robot Exposure is instrumented using robot exposure in Denmark, Italy, Sweden, Finland and France
in the same time periods. Observations are weighted by commuting zone population.
On average, exposure to robots affects both types of workers negatively. However, the magnitudes of
these effects are different for some outcomes and similar for others. In particular, the wage component
of career values is affected similarly for high- and low-skilled workers, with one additional robot per 1000
workers leading to approximately $2.4k reduction in their lifetime income if their career opportunities
were fixed over time. The magnitudes in column (3) for Panel A and Panel B (0.237 vs 0.241) indicate
comparable estimates for those below-college and college-level education, with the same 1% level of
statistical significance. At the same time, the magnitudes in column (4), the component of career value
coming from fixing wages and looking at the change in career path, looks similarly more negative for
the low-skilled workers. For low-skilled workers, the effect is -0.152, while for high-skilled workers it is
equal to -0.104. This implies that for high-skilled, education mitigates the negative effect of robotization,
14To compute career values separately, we split the resumes in the BGT data into 2 groups: those with college and above,
and those with below college education. We compute transition matrices separately for these two groups. We then merge
these data with wages by occupation and state for each group and obtain a measure of career values by education at both
the occupation and the local labor market levels.
26
and high-skilled workers’ lifetime career prospects are less affected. Numerically, these changes imply
that 1 extra robot per 1000 workers leads to approximately $1,520 reduction in lifetime earnings from
upward mobility for low-skilled workers and $1040 reduction in lifetime earnings from upward mobility
for high-skilled workers, and this contrast is even more striking taking into account that average career
values are lower for low-skilled workers.
Overall, the results in Table 6 suggest that robotization mostly affects the career prospects of lowskilled workers, thus widening the gap between high- and low- skilled workers and exacerbating inequality.
Robot Adoption and Inequality in Career Values Here, we further study the implications of
robot adoption for the inequality in career values. More specifically, to compute different measures of
inequality, we take all of our subjects with resumes and align them on the horizontal axis from the
smallest to the largest, thus computing a predicted Lorentz curve. After that, we can compute different
percentile measures of inequality in carer values. We present these results in Table 7. As one can see,
higher robot exposure led to a significant positive change in the inequality in career values between 2000
and 2016, when comparing career values at the 75th percentile to those at the 25th percentile or at the
90th percentile to those at the 50th or 10th percentile. This effect seems to be concentrated in the upper
part of the distribution. Table A.20 in the appendix replicates these results but includes the share of
employment in routine occupations and change exposure to imports from China from 1990 to 2007 as
controls.
Table 7: Robot Adoption and Inequality in Career Values. IV Lasso
Ratio of Career Values in 2016 for Percentiles
50 over 10 75 over 25 90 over 50 90 over 10
(1) (2) (3) (4)
U.S. Robot Exposure ’04-’16 -0.001 0.004*** 0.010*** 0.010***
(0.001) (0.001) (0.003) (0.003)
Mean of D.V. 1.212 1.278 1.309 1.588
F-stat. 0.016 12.888 30.388 22.448
Inequality Measure ’00 Yes Yes Yes Yes
Census Division FE Yes Yes Yes Yes
A&R Controls Yes Yes Yes Yes
Observations 722 722 722 722
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets and
clustered by state. All columns include census division dummies as well as controls for demographic characteristics of CZs
in 1990 identical to those in Acemoglu and Restrepo (2020) (log population; the share of females; the share of population
above 65 years; the shares of population with no college, some college, college and professional degrees, and masters and
doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment in manufacturing; the share
of female employment in manufacturing; and the share of employment in light manufacturing, namely, textile industry
and the paper, publishing, and printing industry). We control for the corresponding inequality measure in 2000 which is
constructed similarly to the dependent variable; it is the ratio of the career values at different percentiles in the year 2000.
U.S. Robot Exposure is instrumented using robot exposure in Denmark, Italy, Sweden, Finland, and France in the same time
periods. Robust standard errors are clustered by state. The dependent variables are the ratios of the career values of different
occupations within a labor market at different percentiles. Observations are weighted by commuting zone population.
27
Heterogeneity with Respect to Sources of Upward Mobility. Another way of studying whether
education indeed helps to mitigate the negative effects of robotization is carrying out a heterogeneity
analysis. In particular, we would like to know whether education and other sources of upward mobility
help to significantly mitigate the negative effect of robotization on the labor markets. To do so, we
estimate the following reduced-form regression equation:
∆yc,(t0,t1) = β1 + β2RbtEUc,(t0,t1) × M bltyc,t0 + β3RbtEUc,(t0,t1) + β4M bltyc,t0 + βXc + ε
`
c,(t0,t1)
, (8)
Here ∆yc,(t0,t1)
is the outcome of interest, RbtEUc,(t0,t1)
is an average robot exposure in EU countries
weighted by commuting zone pre-existing employment in manufacturing, while M bltyc,t0
is a measure of
education or upward mobility, to be specified subsequently. Note that we had to change our empirical
strategy here by switching to a reduced form estimation. It is conceptually similar to the IV specification
but precise, which is important for the case of the interaction term. As an instrument, we use the average
robot exposure in 5 European countries.15 Our results are robust to using different sets of countries.
We estimate equation (8) and present the results in Table 8. We first use school enrollment among
those aged 18 and above as a proxy for the opportunities for upward mobility among the younger population, who are more vulnerable to the effects of limited mobility from a career perspective. 18+ school
enrollment combines college education, delayed high school completion, and vocational training. Column
(1) suggests that career values decline less in places with higher 18+ school enrollment rates, and this
interaction effect is especially pronounced for low-skilled workers (column 2), while it’s markedly smaller
and insignificant for high-skilled workers (column 3). We believe that the result in column (2) is mostly
explained by the presence of community colleges and vocational training programs, which allow affected
workers to adjust their skills to the demands of the market. In columns (4) and (5) we re-estimate
equation (8) using contemporaneous average wages (column 4) or employment (column 5) as dependent
variables. For these outcomes, we no longer see the effect of robotization mitigated. The coefficient of
the interaction term for wages (column 4) is negative and significant, rather than being positive. The
same coefficient for employment (column 5) is positive, though insignificant. These results suggest that
the opportunities to obtain education may increase people’s career prospects, rather than their contemporaneous labor market outcomes. Overall, the results from Table 8 highlight the importance of looking
at career prospect measures like career values when studying the impact of economic changes on labor
market outcomes.
In the appendix, we also repeat the estimation of equation (8) using alternative proxies for opportunities for upward mobility, specifically average level of education (Table A.5) and local intergenerational
mobility (Table A.6).16 The results are qualitatively similar to the results in Table 8.
15By doing so, we follow Acemoglu and Restrepo (2020).
16More specifically, we used neighborhood intergenerational mobility fixed effect from Chetty et al. (2016)
28
Table 8: Automation, Career Values, and 18+ School Enrollment
∆ Career Values 2000-2016 ∆ Wage 2000-2015 ∆ Employment 2000-2015
All Below College College or Higher
(1) (2) (3) (4) (5)
EU Instrument ’04-’16 -0.086*** -0.245*** -0.053** -0.000 -0.049**
(0.021) (0.037) (0.020) (0.001) (0.023)
18+ School Attendance ’00 1.255 -15.315*** 6.374 0.422*** 12.178***
(3.959) (5.588) (4.421) (0.135) (4.446)
EU Instrument ’04-’16 x 18+ School Attendance ’00 0.392* 2.239*** -0.033 -0.013** 0.015
(0.225) (0.357) (0.218) (0.007) (0.240)
Mean of D.V. -0.139 -0.232 -0.262 0.135 0.036
R-squared 0.489 0.257 0.445 0.535 0.751
Census Division FE Yes Yes Yes Yes Yes
A&R Controls Yes Yes Yes Yes Yes
Observations 722 722 722 722 722
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets
and clustered by state. All columns include controls for demographic characteristics of commuting zones in 1990 identical
to those in Acemoglu and Restrepo (2020) (log population; the share of females; the share of population above 65 years; the
shares of population with no college, some college, college and professional degrees, and masters and doctoral degrees; the
shares of Whites, Blacks, Hispanics, and Asians; the share of employment in manufacturing; the share of female employment
in manufacturing; and the share of employment in light manufacturing, namely, textile industry and the paper, publishing,
and printing industry). The EU instrument is the average robot exposure in Denmark, France, Finland, UK and Germany
during 2004-2016. Observations are weighted by commuting zone population.
5 Mechanisms
In this section, we look into mechanisms behind the main findings, as summarized in Tables 3, 4, 5, and
6. Our results so far suggest that expected lifetime incomes and upward mobility of workers declined
as a result of automation, and more so for low-skilled workers. However, automation effects could be so
noticeable and pronounced due to a number of reasons. It could happen because of spillover effects to
other industries, especially non-tradable ones (supply-side effects), or general equilibrium changes in the
patterns of consumption and consumer expectations (demand-side effects). In this section, we summarize
some evidence pointing towards the mechanisms. We furthermore look at whether other sources of
mobility mitigate the negative effects of automation and robotization on career values and compare it
with the effect on current employment and wages.
5.1 Spillovers from Manufacturing to Other Industries
Our focus on career values makes the interdependence between different industries in a given location
more visible. Table 9 summarizes how exposure to robots affects career values in different industry
groups. We combine all manufacturing, trade, and service NAICS categories to simplify exposition. The
coefficients are all statistically significant and close in magnitudes. This happens since career values
look at all employment options workers have, instead of focusing only on their current industry. In fact,
different industries are affected differentially by automation shocks (see e.g., Table A.9 or Table A.10 in
the appendix), but the variation across industries in these shocks is attenuated from the perspective of
29
workers when one takes into account all possible career transitions.17 18
Table 9: Exposure to Robots and Change in Career Values, by Industry Group. IV Lasso
Manufacturing Retail and Wholesale Services Management Other Industries
(1) (2) (3) (4) (5)
U.S. Robot Exposure ’04-’16 -0.354*** -0.346*** -0.415*** -0.232** -0.389***
(0.043) (0.038) (0.040) (0.097) (0.039)
Mean of D.V. -0.235 -0.405 -0.390 0.348 -0.138
F-stat. 93.875 80.315 103.167 7.348 106.558
Census Division FE Yes Yes Yes Yes Yes
A&R Controls Yes Yes Yes Yes Yes
Observations 722 722 722 616 722
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets
and clustered by state. All columns include census division dummies as well as controls for demographic characteristics
of commuting zones in 1990 identical to those in Acemoglu and Restrepo (2020) (log population; the share of females;
the share of population above 65 years; the shares of population with no college, some college, college and professional
degrees, and masters and doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment
in manufacturing; the share of female employment in manufacturing; and the share of employment in light manufacturing,
namely, textile industry and the paper, publishing, and printing industry). U.S. Robot Exposure is instrumented using robot
exposure in Denmark, Italy, Sweden, Finland, France, Germany, Spain and the UK in the same time periods. Industrial
classification is based on 2-digit NAICS. Observations are weighted by commuting zone population.
5.2 Consumer Responses
In this subsection, we look at whether automation had any impact on consumption and economic expectations. In particular, we tested whether automation exposure results in a change in average household
spending on groceries and household items. To this end, we use the Nielsen Consumer Panel data on
household-level consumption. The earliest data in this panel go back to 2004, and we use data from
households common to both the 2004 and 2016 panels to control for unobserved household characteristics. In Table 10, we present the IV specifications, regressing the logged difference between 2016 and
2004 aggregate household spending, and adjusting for inflation. Running specification 5 with the change
in total household spending on grocery and household items shows that, with greater exposure to robotization, household consumption saw a greater and statistically significant decline. The magnitude of
the effect implies that one additional robot per 1000 workers resulted in a 2.5% decline in the overall
consumption between 2004 and 2016, and more so in high-manufacturing commuting zones. Table A.21
in the appendix replicates these results but includes the share of employment in routine occupations and
change in exposure to imports from China from 1990 to 2007 as controls, and provides practically the
same results.
17Respective results for wages are summarized in Tables A.11 and A.12
18In the online appendix section A.9, we also check the patterns of business creation and destruction in different sectors.
We follow Mian and Sufi (2014), who use 4-digit NAICS codes to define tradable vs. non-tradable establishments, to estimate
this effect. We find a negative but insignificant direction on the number of establishments in places with higher exposure to
robots in the construction sector. Non-tradable establishments in regions with higher robot exposure also have a significantly
smaller number of employees. Overall, these results are suggestive of some negative effects of automation on local business
growth, consistent with declining demand for goods and services.
30
Table 10: Automation Exposure and Household Retail Consumption. IV Lasso
Log (∆ Total Household Spending) Log (∆ Total Spending)
All High Manuf. CZ Low Manuf. CZ
(1) (2) (3)
U.S. Robot Exposure ’04-’16 -0.025*** -0.036*** -0.017
(0.004) (0.003) (0.041)
Mean of D.V. -0.300 -0.309 -0.290
F-stat. 14.100 11.407 0.003
Census Division FE Yes Yes Yes
A&R Controls Yes Yes Yes
Observations 11146 5620 5498
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets
and clustered by state. The source of household spending data is NielsenIQ Consumer Retail Panel, made accessible by the
Kilts Center at the University of Chicago. Panels are from 2004 and 2016. The outcome compared the log of the difference
in household spending. The analysis is based on the households that are common between 2004 and 2016, and uses the
difference in spending of each household. Household spending in inflation-adjusted. All columns include census division
dummies as well as controls for demographic characteristics of commuting zones in 1990 identical to those in Acemoglu and
Restrepo (2020) (log population; the share of females; the share of population above 65 years; the shares of population with
no college, some college, college and professional degrees, and masters and doctoral degrees; the shares of Whites, Blacks,
Hispanics, and Asians; the share of employment in manufacturing; the share of female employment in manufacturing; and
the share of employment in light manufacturing, namely, textile industry and the paper, publishing, and printing industry).
U.S. Robot Exposure is instrumented using robot exposure in Denmark, Italy, Sweden, Finland, France, Germany, Spain
and the UK in the same time periods. Commuting zones are classified as high manufacturing zones if the share of population
employed in the manufacturing sector is greater than the median share of employment in manufacturing in the country. Low
manufacturing zones have below median share of employment in manufacturing. Observations are weighted by commuting
zone population.
Next, we report what happens to people’s economic expectations. In Table 11, we check if greater
exposure to robotization altered people’s perceptions of the economic conditions in any way. As a proxy
for the perception of the economy, we use a question from the Gallup Poll Social Series (GPSS) surveys
measuring economic confidence/economic outlook: “Right now, do you think that economic conditions in
the country as a whole are getting better or getting worse?” and the possible answers are Getting Better;
Getting Worse; Same; Don’t Know; Refused to answer. We code two variables, one is coded better = 1 if
the answer is “better” and 0 otherwise, and the other variable is coded worse = 1 if the answer is “worse”
and 0 otherwise. The response to the survey question employed as an outcome was recorded in 2016. In
local markets that have experienced greater robot exposure, individuals are less likely to state that the
economy is getting better, suggesting that the worsening of people’s perceptions correlates with greater
robotization. These results, furthermore, imply that automation and robotization can affect people’s
long-term decisions, as examined in Section 6.
6 Career Value and Long-term Outcomes
Results in sections 4 and 5 suggest that automation affects future mobility, expected lifetime incomes,
and economic expectations. These changed expectations can, in turn, affect people’s long-term decisions.
In this section, we aim to estimate the causal effects of expected career values, our measure of expected
lifetime income, on schooling, housing, and voting.
31
Table 11: Automation and Perception of the Economy, Gallup. IV Lasso
(1) (2) (3) (4)
Economy has Economy has Economy has Economy has
gotten better gotten better gotten worse gotten worse
U.S. Robot Exposure ’04-’16 -0.004*** -0.004*** -0.001 -0.000
[0.001] [0.001] [0.001] [0.001]
Mean of D.V. 0.04 0.04 0.05 0.05
F-stat. 639 632 639 632
Census Division FE Yes Yes Yes Yes
A&R Controls Yes Yes Yes Yes
Individual Controls No Yes No Yes
Observations 10,165 9,960 10,165 9,960
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets
and clustered by commuting zone. An observation is an individual in 2016. Data on perceptions are from the Gallup Poll
Social Series (GPSS) surveys from 2016. All columns include census division dummies as well as controls for demographic
characteristics of commuting zones in 1990 identical to those in Acemoglu and Restrepo (2020) (log population; the share of
females; the share of population above 65 years; the shares of population with no college, some college, college and professional
degrees, and masters and doctoral degrees; and the shares of Whites, Blacks, Hispanics, and Asians). Columns (2) and (4)
include individual-level controls for gender, age group, educational status, hispanic origin, marital status, employment status
and religious preferences. U.S. Robot Exposure is instrumented using robot exposure in Denmark, Italy, Sweden, Finland,
France, Germany, Spain and the UK in the same time periods. Observations are weighted by commuting zone population
and the survey weight.
6.1 Estimating the effect of career value changes
Assessing the effect of a change in career values on these outcomes requires a different identification
strategy than the one we introduced in Section 4. Here, we employ an additional instrumental variables
strategy to identify the effect of career values on college education, housing permits, and voting. Specifically, we instrument the average market career values of a CZ using the career values from sufficiently
similar but distant communities. We focus on the communities that are in states outside a 100-mile
radius around the commuting zone of interest. That is, we are instrumenting for local career values using
information from outside a ‘donut’ around the area of interest that is plausibly exogenous with regard to
idiosyncratic local labor market shocks.
For each occupation-by-CZ cell, we compute the unweighted mean C
c+100m
o,s,t
V
of the predicted career
values for the same occupation in sufficiently distant states. At the next step, using the local employment
shares in each CZ, we aggregate these predicted career values into predicted labor market career values
V
LMCVc+100m,t for each CZ that use only wage levels and the resulting career values in other states’
commuting zones.
The first-stage equation is then given by:
∆LMCVc,t1,t0 = α0 + α1∆
V
LMCVc+100m,t1,t0 + α2Xt0 + εt1t0
where ∆LMCV c,t1,t0
is the actual change in the local market career value of CZ c between year t0 and t1.
We use the predicted local market career values in the 2SLS estimation for long-term outcomes and find
that an increase in career values is predictive of investment in college education, housing, and voting, as
32
detailed next.
6.2 Implications of career value changes
The decline in career values in a region can, on the one hand, motivate a desire to re-skill or obtain
further training to start at a better–paying occupation and recover expected losses. On the other hand,
better opportunities for upward transitions and better returns to education overall can increase people’s
willingness to study in places with higher career values. To test which of these effects prevails, in what
follows, we summarize how career values affect people’s average levels of education.
Next, we look at how career values affect housing investment. Expectations about future employment
and earnings may influence long-term financial investment decisions, which can influence household demand for housing. Moreover, higher career values might affect beliefs about future housing demand. In
what follows, we look at the logged housing construction permits times 100 per capita (based on 2000
population) and investigate the association to career values.
Finally, we also take a look at political behavior by checking for shifts in voting. The decline in career
values and, overall, worsened opportunities for upward mobility can lead to a higher probability of voting
for populist candidates. For instance, Panunzi et al. (2020) argue that unmatched expectations make
people more risk-loving and more willing to support populist candidates like Donald Trump. Furthermore,
Guiso et al. (2017) and Di Tella and Rotemberg (2018) argue that income loss and feelings of betrayal
can generate anti-elite preferences and, as a consequence, higher support for more populist candidates.
Table 12 summarizes these results, with the first stage reported in column (1). As one can see in
columns (3), (5), and (7), career values have significant effects on long-term decisions in the predicted
direction, for IV Lasso estimates. Columns (2), (4), (6) report the results for OLS estimation for comparison. Numerically, higher career values led to higher shares of people getting higher education over the
corresponding time period, with one standard deviation of career values leading to 1.1 p.p. increase in the
share of those getting higher education. In a similar vein, the coefficient in column (4) implies that one
standard deviation of career values led, on average, to a decline of 379.6 in the number of new permits,
which constitutes 13.2% of the initial number of permits in 2004.19 Columns (6) and (7) demonstrate
that in regions where the local market career values, instrumented by the average market career values
of nearby CZz, were lower, Trump vote shares were higher in the 2016 Presidential Election, controlling
for the GOP vote share of the 2012 Presidential Election. Numerically, one standard deviation increase
in career values led to a 0.68 p.p. decrease in Trump’s vote share in 2016. We will focus on this outcome
more closely in the following section.
19To get this number, we multiplied the coefficient, 6.56 with one standard deviation of career values change, 1.51.
Noting that all housing permits are divided by the population of the CZ and multiplied by 10,000, we derive the number
6.56*1.51*381,634/10,000= 379.6.
33
Table 12: Career Value Effects on College Education, Housing Permits, and Voting
∆ LMCV, 04-16 ∆ College % 00-15 ∆ ln Housing Permits GOP Vote Share 2016
First-Stage OLS IV OLS IV OLS IV
(1) (2) (3) (4) (5) (6) (7)
Predicted ∆ LMCV ’04-’16 3.462***
(0.308)
∆ LMCV ’04-’16 0.297*** 0.719*** 6.563*** 13.745*** -0.055 -0.448**
(0.068) (0.138) (1.987) (4.562) (0.128) (0.228)
Mean of D.V. -0.139 4.805 4.805 -60.920 -60.920 65.368 65.368
KP F-stat. 126.407 124.970 127.464
R-Squared 0.564 0.585 0.522 0.358 0.335 0.982 0.982
Census Division FE Yes Yes Yes Yes Yes Yes Yes
A&R Controls Yes Yes Yes Yes Yes Yes Yes
GOP Vote Share 2012 Yes Yes
Observations 722 722 722 669 669 722 722
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets
and clustered by state. All columns include census division dummies as well as controls for demographic characteristics
of commuting zones in 1990 identical to those in Acemoglu and Restrepo (2020) (log population; the share of females;
the share of population above 65 years; the shares of population with no college, some college, college and professional
degrees, and masters and doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment
in manufacturing; the share of female employment in manufacturing; and the share of employment in light manufacturing,
namely, textile industry and the paper, publishing, and printing industry). Columns (6) and (7) also include the share of
votes won by the Republican candidate in the 2012 Presidential elections as a control. The change in local market career
values is instrumented using the mean change in local market career values in commuting zones farther than 100 miles.
Observations are weighted by commuting zone population.
7 Automation, Career Values, and Political Outcomes
The results in Table 12 (columns 6 and 7) suggest that career prospects may also have effects on political outcomes. In this section, we investigate how both economic transformation and job opportunities
may be affecting the political landscape. The theoretical literature starting with Meltzer and Richard
(1981) suggests that individual income should affect preferences for redistribution. Then, if the income
of a median voter goes down, which happens when inequality goes up, that should increase the support
of left-wing candidates and parties. At the same time, economic transformations of recent years also
changed the political structure, with new candidates/parties and candidates with a new, populist agenda
becoming increasingly popular, with a more traditional left-to-right cleavage becoming less salient. Autor
et al. (2020a), Autor et al. (2016), and Colantone and Stanig (2018b) suggest that globalization plays
an important role in the rise of populism support, while refugee migration also is becoming an increasingly important issue (Dustmann et al. (2018), Halla et al. (2017), (Noury and Roland, 2020)). Anelli
et al. (2021), Gallego et al. (2022), and Frey et al. (2018) go one step further and study the impact of
robotization on political preferences, while Upward and Wright (2023) study the impact of individual job
loss.
Our focus on career values allows us to enrich this analysis. We contribute to the literature in several
ways. First, we focus on the impact of career prospects in parallel with the study of automation, thus
being able to understand how many of the results are driven by forward-looking expectations. Second,
we study if heterogeneity analysis is consistent with economic transformation driving political outcomes
34
and if economic shocks parallel changes in political preferences. Third, we also focus on the supply-side
factors, by looking at campaign visits and policy positions.
Theoretically, there are reasons why economic changes can lead to more populist votes. In theory,
negative economic shocks can lead to the support of extreme left or extreme right politicians (Margalit,
2019). Guiso et al. (2017) and Di Tella and Rotemberg (2018) argue that income loss and the feeling of
betrayal can generate anti-elite preferences, thus leading to more populist voting. Panunzi et al. (2020)
argue that it can make people unhappy about their income and make them more risk-loving, thus also
increasing the support for populism. Our preliminary results from Table 12 suggest that career prospects
affected voting for Trump, arguably the most populist out of all general election presidential candidates
in the United States.
More generally, this part of the analysis contributes to the literature on the drivers of populism,
recently reviewed by Guriev and Papaioannou (2022). Recent literature suggests that the economic crises
(e.g., the Great Recession), and more generally the negative economic shocks, have contributed to the
popularity of populism (e.g., Eichengreen, 2018; Rodrik, 2018; Autor et al., 2020b; Colantone and Stanig,
2018a; Guiso et al., 2019, 2020; Dippel et al., 2017; Fetzer, 2019; Anelli et al., 2019, etc.). Moreover, low
levels of trust, identity politics, and growing hostility towards immigrants also contributed to the growth
of populism (Algan and Cahuc, 2013; Dustmann et al., 2017; Gennaioli and Tabellini, 2023), as did the
internet and social media (Guriev et al., 2021; Manacorda et al., 2022). We contribute to this literature
by focusing on forward-looking measures of people’s welfare and by exploring the mechanisms behind the
results.
7.1 Automation and Political Preferences
In this subsection, we look at the impact of automation on the vote shares of Republican candidates for
various years, obtained from the Election Atlas. We use the IV Lasso specification, as outlined above
(equation (5)). In Table 13, we find that robotization positively affected the vote share of Trump in 2016
(column 1, the coefficient is positive and significant at 1% level), but not the vote shares of Romney and
McCain, the Republican candidates in the previous general elections. The point estimate implies that
one extra robot per 1000 workers led to a 0.38 p.p. increase in the vote share of Trump.20
In columns (4)-(6) of the table, we report what happens if voting is predicted by local labor market
career values, again using commuting zones beyond 100 miles circle to instrument for local career values.
Column (4) (career values and voting for Trump) reproduces the results from column (7) of Table 12,
while the effect of career values on voting for McCain and Romney in 2008 and 2012, respectively, is
numerically smaller and statistically insignificant. Numerically, one standard deviation increase in career
value led to a 0.68 p.p. increase in Trump vote share in a given commuting zone; for earlier years, we
can rule out the effects larger than 0.41 p.p. for 2012 and larger than 0.30 p.p. for 2008. These findings
are consistent with the idea that economic shocks strengthen voting for extreme candidates (Margalit,
20Note that we can extend the robot exposure part of the table back to 1988 in online appendix Table A.17, where the
effect of exposure on Republican Presidential candidate vote share is insignificant in each year.
35
Table 13: Vote Shares and Automation Shocks, 2008-2016, IV Lasso
Voter Share of Republican Presidential Candidate
2016 2012 2008 2016 2012 2008
(1) (2) (3) (4) (5) (6)
U.S. Robot Exposure ’04-’16 0.380*** 0.039 -0.022
(0.104) (0.130) (0.139)
∆ LMCV ’04-’16 -0.448** -0.106 0.081
(0.228) (0.158) (0.195)
Mean of D.V. 65.368 60.204 57.053 65.368 60.204 57.053
F-stat. 16.794 0.069 0.617 7.398 0.613 0.171
Census Division FE Yes Yes Yes Yes Yes Yes
A&R Controls Yes Yes Yes Yes Yes Yes
Observations 722 722 722 722 722 722
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets and
clustered by commuting zone. All columns include census division dummies, demographic characteristics of commuting zones
in 1990 identical to those in Acemoglu and Restrepo (2020) (log population; the share of females; the share of population
above 65 years; the shares of population with no college, some college, college and professional degrees, and masters and
doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment in manufacturing; the share
of female employment in manufacturing; and the share of employment in light manufacturing, namely, textile industry and
the paper, publishing, and printing industry). U.S. Robot Exposure is instrumented using robot exposure in Denmark, Italy,
Sweden, Finland, France, Germany, Spain and the UK in the same time periods. The change in local market career values is
instrumented using the mean change in local market career values in commuting zones farther than 100 miles. Observations
are weighted by the commuting zone population.
2019). Further analysis in this section is aimed at understanding a more nuanced picture of what’s going
on.
7.2 Distributional Effects of Automation and Voting Preferences
The hypothesis that economic changes affect voting is almost as old as the modern political economy
literature. However, this hypothesis implies important heterogeneity: people who are the most affected
by the shock should be people who change their voting behavior in response. This is a sanity check
that the literature rarely tests for. Our data allow us to carry out this heterogeneity analysis, looking
in parallel at the distributional effects of robotization and voting intentions. Put differently, we study
the distributional effects of robot exposure on economic outcomes and then compare them against the
distributional effects of robot exposure on electoral outcomes.
For example, Table 6 suggests that low-skilled people are more likely to be affected by robot exposure,
more so in high manufacturing commuting zones. In Table A.13, we look at the distributional effects of
robotization employment. We confirm that even for contemporary effects, the low-skilled are more likely
to be affected (and those with education at the master’s level and above were placed into more jobs in
manufacturing, as column (8) of Table A.13, Panel A implies).
Now, we study whether those who are most likely to be affected are also more likely to vote for
Trump. Figure 8 shows the shares of self-reported Trump voters by their levels of education. We find
that in parallel to the results in Table 6 and Table A.13, the least educated people had the highest share
36
of Trump voters. Furthermore, as Figure 8 suggests, people from the highest quintile of manufacturing
employment were also the most likely Trump voters. Thus, heterogeneity analysis with respect to education is consistent with the idea that the least educated voters supported Trump and likely suffered from
the negative economic consequences of robotization.21
Figure 8: Voting Intentions by Education and Share of Manufacturing, 2016 Presidential Elections
(a) By Educational Attainment (b) By Share of Employment in Manufacturing
Notes. The source of data is the Congressional Cooperative Election Survey (2016). Commuting zones are classified by the
percentile into which the share of employment in manufacturing in 1990 falls.
Figure 9: Trump voting Intentions, Predicted Family Income, and Age Family Income, as Predicted by EU Automation -.12 -.1 -.08-.06-.04-.02018-2526-4041-5556+Voting for TrumpNotes. Robust standard errors are clustered at the commuting zone level. Census division dummies are included. The
source of data is the Congressional Cooperative Election Survey (2016). Household income is predicted by robot exposure
in Denmark, Italy, Sweden, Finland, France, Germany, Spain and the UK.
21In the appendix, we repeat this analysis by race. We cannot estimate career values by race, but we can look at
employment. We report CCES-based Trump vote intentions by race in Figure A.9. By comparing the columns in Figure
A.9, we conclude that African American and Hispanic voters support Trump less than their economic situation predicts.
White low-skilled voting behavior in this election is, thus, well explained by economic logic. At the same time, the behavior
of African-American and Hispanic voters is not mostly driven by their economic interests. In other words, for people of
other races, their economic situation was less important for their decision to vote for a particular candidate in the 2016
Presidential elections. These findings are consistent with the studies on sociodemographic drivers of populist voting based
on OLS estimates from electoral survey data (Inglehart and Norris, 2016; Mocan and Raschke, 2016; Serani, 2016).
37
7.3 Automation, Income, Age, and Voting
In this subsection, we ask whether the link between automation and income shocks is stronger for younger
voters. More specifically, we estimate vote choice as a function of household income and demographic
characteristics, with household income instrumented by CZ’s predicted exposure to automation, using
EU industry-level automation to construct the instrument. Dependent variable data comes from the
Congressional Cooperative Election Survey. These results are reported in Figure 9. The figure suggests
that the association between income and voting for Trump seems to be numerically strongest for the
youngest people, while, at the same time, it becomes numerically smaller and statistically insignificant
for 56+. This (albeit suggestive) evidence strengthens our interpretation of voting for Trump as partly
an economic phenomenon. Age heterogeneity is consistent with a career value-based explanation: income
is a less important predictor of voting for the oldest workers, for whom most of their career transitions
already occurred in the past, and a more important predictor of voting for the youngest voters, who
should care the most about their career values and associated opportunities for upward mobility.
7.4 Automation and the Supply-Side of Politics
In this subsection, we study whether increased demand for populism triggered some changes in the
behavior of politicians.
Table 14: Automation and Campaign Visits, 2016. IV Lasso
Republicans Democrats
All November 2016 October 2016 All November 2016 October 2016
(1) (2) (3) (4) (5) (6)
U.S. Robot Exposure ’04-’16 1.129 0.376* 0.818* -0.917 -0.532 -0.289
(1.021) (0.207) (0.423) (1.609) (0.366) (0.689)
Mean of D.V. 1.924 0.293 0.823 1.708 0.334 0.655
KP F-stat. 1.072 1.780 4.332 0.605 2.194 0.411
Observations 722 722 722 722 722 722
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets
and clustered by state. All columns include census division dummies as well as controls for demographic characteristics
of commuting zones in 1990 identical to those in Acemoglu and Restrepo (2020) (log population; the share of females;
the share of population above 65 years; the shares of population with no college, some college, college and professional
degrees, and masters and doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment
in manufacturing; the share of female employment in manufacturing; and the share of employment in light manufacturing,
namely, textile industry and the paper, publishing, and printing industry). U.S. Robot Exposure is instrumented using
robot exposure in Denmark, Italy, Sweden, Finland, France, Germany and UK in the same time periods. Observations are
weighted by commuting zone population.
In Table 14, we check if the automation exposure of a CZ is correlated with the number of campaign
visits to that location, looking separately at the visits by the Republican and Democratic Presidential
candidates while campaigning for the 2016 Presidential Election.22 In panel A, we find no significant
evidence of higher campaign visits to CZs with higher automation exposure by the Democratic candidates.
Still, we find a positive association for the Republican candidate. One standard deviation increase in
22The source of campaign visit data is Devine (2018). We were not able to obtain similar data for 2012 or 2008.
38
robot exposure is associated with 0.81 additional visits to a CZ for the Republican candidate in October
2016 and 0.37 additional visits in November. 23
8 Conclusion
In this study, we document changes in the growth of career values between 2000 and 2016. Career values
are a projection of one’s ability to move across occupations and earning potential, and as such, capture
a longer term view of one’s economic future. We find that the average local labor market career value
grew by $2,100 between 2000 and 2008, which corresponds to an 0.9% increase relative to the average
expected lifetime career value in 2000, compared to a $200 or a 0.09% decline in the local market career
values between 2008 and 2016. Wage growth slows on average, but an important factor in the stalling of
the growth of career values is reduced upward occupational mobility. The reduction in career value due
to the reduced upward mobility (holding wages constant) was as large as $12,500 between 2000 and 2016.
We test and show that automation and robotization are contributing factors to the reduced growth
of career values. We find that a one unit increase in the number of robots per 1000 workers decreases
career values by $3,820, or 1.7% of the average career values in 2000. The negative effects of automation
are concentrated in the manufacturing-intensive local markets. Low-skilled workers suffered more from
robot adoption, with their opportunities for upward mobility declining the most.
Our analysis also points to a possible spillover effect of automation: We find that in local labor markets
with higher automation exposure, employment shares and wages decline in a multitude of industries,
including those that are not directly facing automation themselves. We find lower employment or a wage
decline for industries such as retail, wholesale trade, healthcare, construction, professional and technical
services. The negative labor market outcomes observed in these industries could be because workers whose
own industries are impacted by automation may have to or want to migrate to other industries, increasing
competition, particularly among less-than-college educated workers. Increased supply of workers may
reduce their bargaining power for wages and result in a decline in wages if the demand for workers in
non-manufacturing industries cannot accommodate these workers. However, increasing supply does not
necessarily imply a decline in employment rates in other industries. Therefore, displacement of workers in
manufacturing jobs and their migration to jobs in other industries is not sufficient to explain all observed
patterns in the data.
We focus on a number of possible explanations for the decline in career values, consistent with the
spillover effects of automation. A second channel through which automation can influence career values
23We also ask whether exposure to automation and robotization translated into differential policies. Table A.22 presents
roll call votes by politicians as well as the Nokken and Poole scores for the US House of Representatives for the year 2016
(Nokken and Poole, 2004; Lewis et al., 2023). Here, a score of -1 indicates a far-left ideology and a score of 1 indicates
a far-right ideology, and 0 represents moderate ideologies. Panel A in the table finds no correlation on average between a
CZ’s robot exposure and the score of the candidates representing it in the House of Representatives. Panels B and C break
down the CZs by high and low manufacturing labor shares. We see that in CZs with higher manufacturing intensity, greater
exposure to automation is associated with a more conservative record of the politician representing the CZ, whereas there
is no statistically significant relationship in low manufacturing-intensity regions. Specifically, in high manufacturing areas,
a one-unit increase in automation exposure of labor is associated with a 0.026 p.p. increase in the roll call score, controlling
for the score in 2012. This implies a more conservative roll-call voting behavior.
39
and result in declines in employment and wages is through reduced spending and long-term investments
in local markets. If future career value expectations decline in regions which have been more exposed to
automation, the spending in items such as retail and hospitality industry may decline. We find that higher
exposure to automation results in an increase in the new permits issued for traditional and manufacturing
establishments, but there is also a decline in the number of employees per establishment. We see that
locally-consumed industries suffer from automation whereas others (e.g., mining, utilities) do not.
Finally, we document several changes in the political area. We show that both robot adoption and
expected career values affect voting for Trump but not for other Republican Presidential candidates. We
find that the heterogeneity of the effect of robotization by education and age is consistent with career
value explanation for voting. We furthermore document changes in the campaign visits and policy voting,
i.e., that the supply side of politics also responds to the changes in political demands.
In recent years, artificial intelligence (AI) technologies were also on the rise. Our identification strategy
does not allow us to study the effects of AI precisely. However, based on what we learn about robots, we
hypothesize that as some white-collar jobs get replaced by AI, it would exacerbate inequality, diminish
upward transitions due to reduced demand for middle level managers, and erode the middle class.
Overall, our findings highlight that it is essential to think about the effects of automation from a
broader perspective—focusing on industries beyond those that are most immediately impacted and beyond the immediate occupation of workers. Automation and robotization affect both workers’ wages and
occupational mobility, and expectations regarding future wages and mobility may influence individuals’
long-term investments, perceptions, and political behaviors today. Our study documents these effects.
References
Acemoglu, D., S. Johnson, and J. A. Robinson (2001, December). The colonial origins of comparative
development: An empirical investigation. American Economic Review 91 (5), 1369–1401.
Acemoglu, D. and P. Restrepo (2019). Automation and new tasks: how technology displaces and reinstates
labor. Journal of Economic Perspectives 33 (2), 3–30.
Acemoglu, D. and P. Restrepo (2020). Robots and jobs: Evidence from US labor markets. Journal of
Political Economy 128 (6), 2188–2244.
Acemoglu, D., J. A. Robinson, and R. Torvik (2013). Why do voters dismantle checks and balances?
Review of Economic Studies 80 (3), 845–875.
Alekseeva, L., J. Azar, M. Gine, S. Samila, and B. Taska (2021). The demand for AI skills in the labor
market. Labour Economics 71, 102002.
Algan, Y. and P. Cahuc (2013). Trust, institutions and economic development. Handbook of Economic
Growth 1.
Algan, Y., S. Guriev, E. Papaioannou, and E. Passari (2017). The European trust crisis and the rise of
populism. Brookings Papers on Economic Activity 2017 (2), 309–400.
Alvaredo, F., D. Cogneau, and T. Piketty (2021). Income inequality under colonial rule. evidence from
French Algeria, Cameroon, Tunisia, and Vietnam and comparisons with British colonies 1920–1960.
Journal of Development Economics 152, 102680.
Anelli, M., I. Colantone, and P. Stanig (2019). We were the robots: Automation and voting behavior in
Western Europe. BAFFI CAREFIN Centre Research Paper (2019-115).
40
Anelli, M., I. Colantone, and P. Stanig (2021). Individual vulnerability to industrial robot adoption
increases support for the radical right. Proceedings of the National Academy of Sciences 118 (47),
e2111611118.
Antràs, P., A. de Gortari, and O. Itskhoki (2017, 2017). Globalization, inequality and welfare. Journal
of International Economics 108, 387–412.
Arntz, M., T. Gregory, and U. Zierahn (2017). Revisiting the risk of automation. Economics Letters 159,
157–160.
Autor, D., D. Dorn, G. Hanson, and K. Majlesi (2016). A note on the effect of rising trade exposure on
the 2016 presidential election. Unpublished Manuscript.
Autor, D., D. Dorn, G. Hanson, and K. Majlesi (2020a, October). Importing political polarization? the
electoral consequences of rising trade exposure. American Economic Review 110 (10), 3139–83.
Autor, D., D. Dorn, G. Hanson, and K. Majlesi (2020b). Importing political polarization? The electoral
consequences of rising trade exposure. American Economic Review 110 (10), 3139–3183.
Autor, D. H., D. Dorn, and G. H. Hanson (2013). The China syndrome: Local labor market effects of
import competition in the United States. American Economic Review 103 (6), 2121–68.
Autor, D. H., F. Levy, and R. J. Murnane (2003). The skill content of recent technological change: An
empirical exploration. The Quarterly journal of economics 118 (4), 1279–1333.
Battisti, M., C. Dustmann, and U. Schoenberg (2023, 02). Technological and organizational change and
the careers of workers. Journal of the European Economic Association 21 (4), 1551–1594.
Bilal, A. and E. Rossi-Hansberg (2021). Location as an asset. Econometrica 89 (5), 2459–2495.
Black, S. E. and P. J. Devereux (2011). Recent developments in intergenerational mobility. Handbook of
Labor Economics 4, 1487–1541.
Che, Y., Y. Lu, J. R. Pierce, P. K. Schott, and Z. Tao (2022). Did trade liberalization with China
influence US elections? Journal of International Economics 139, 103652.
Chernozhukov, V., C. Hansen, and M. Spindler (2015). Post-selection and post-regularization inference
in linear models with many controls and instruments. American Economic Review 105 (5), 486–90.
Chetty, R., D. Grusky, M. Hell, N. Hendren, R. Manduca, and J. Narang (2017). The fading American
dream: Trends in absolute income mobility since 1940. Science 356 (6336), 398–406.
Chetty, R. and N. Hendren (2018). The impacts of neighborhoods on intergenerational mobility I: Childhood exposure effects. The Quarterly Journal of Economics 133 (3), 1107–1162.
Chetty, R., N. Hendren, and L. F. Katz (2016). The effects of exposure to better neighborhoods on children: New evidence from the moving to opportunity experiment. American Economic Review 106 (4),
855–902.
Chetty, R., N. Hendren, P. Kline, and E. Saez (2014). Where is the land of opportunity? The geography
of intergenerational mobility in the United States. The Quarterly Journal of Economics 129 (4), 1553–
1623.
Cockriel, W. M. (2023). Machines eating men: Shoemakers and their children after the mckay stitcher.
Technical report, University of Chicago.
Colantone, I. and P. Stanig (2018a). Global competition and Brexit. American Political Science Review 112 (2), 201–218.
Colantone, I. and P. Stanig (2018b). The trade origins of economic nationalism: Import competition and
voting behavior in Western Europe. American Journal of Political Science 62 (4), 936–953.
Coller, M. and M. B. Williams (1999). Eliciting individual discount rates. Experimental Economics 2,
107–127.
Di Tella, R. and J. J. Rotemberg (2018). Populism and the return of the “paranoid style”: Some
evidence and a simple model of demand for incompetence as insurance against elite betrayal. Journal
41
of Comparative Economics 46 (4), 988–1005.
Dippel, C., R. Gold, and S. Heblich (2015). Globalization and its (dis-) content: Trade shocks and voting
behavior. NBER Working Paper (w21812).
Dippel, C., R. Gold, S. Heblich, and R. Pinto (2017). Instrumental variables and causal mechanisms:
Unpacking the effect of trade on workers and voters. Technical report, National Bureau of Economic
Research.
Dix-Carneiro, R. (2014). Trade liberalization and labor market dynamics. Econometrica 82 (3), 825–885.
Dornbusch, R. and S. Edwards (1991). The macroeconomics of populism. In The macroeconomics of
populism in Latin America, pp. 7–13. University of Chicago Press.
Drake, P. W. (1982). Conclusion: Requiem for populism? Latin American Populism in Comparative
Perspective (Albuquerque: University of New Mexico Press), 218.
Dustmann, C., B. Eichengreen, S. Otten, A. Sapir, G. Tabellini, and G. Zoega (2017). Europe’s trust
deficit. Causes and Remedies. London: Centre for Economic Policy Research.
Dustmann, C., B. Fitzenberger, and M. Zimmermann (2021, 12). Housing Expenditure and Income
Inequality. The Economic Journal 132 (645), 1709–1736.
Dustmann, C., K. Vasiljeva, and A. Piil Damm (2018). Refugee migration and electoral outcomes. The
Review of Economic Studies 86 (5), 2035–2091.
Eichengreen, B. (2018). The populist temptation: Economic grievance and political reaction in the modern
era. Oxford University Press.
Enke, B. (2020). Moral values and voting. Journal of Political Economy 128 (10), 3679–3729.
Faber, M., A. Sarto, and M. Tabellini (2019). The impact of technology and trade on migration: Evidence
from the US. Harvard Business School BGIE Unit Working Paper (20-071).
Fetzer, T. (2019). Did austerity cause Brexit? American Economic Review 109 (11), 3849–3886.
Frank, M. R., D. Autor, J. E. Bessen, E. Brynjolfsson, M. Cebrian, D. J. Deming, M. Feldman, M. Groh,
J. Lobo, E. Moro, et al. (2019). Toward understanding the impact of artificial intelligence on labor.
Proceedings of the National Academy of Sciences 116 (14), 6531–6539.
Frey, C. B., T. Berger, and C. Chen (2018, 07). Political machinery: did robots swing the 2016 US
presidential election? Oxford Review of Economic Policy 34 (3), 418–442.
Frey, C. B. and M. A. Osborne (2017). The future of employment: How susceptible are jobs to computerisation? Technological forecasting and social change 114, 254–280.
Furman, J. and R. Seamans (2019). AI and the economy. Innovation Policy and the Economy 19 (1),
161–191.
Gallego, A., T. Kurer, and N. Schöll (2022). Neither left behind nor superstar: Ordinary winners of
digitalization at the ballot box. The Journal of Politics 84 (1), 418–436.
Gennaioli, N. and G. Tabellini (2023). Identity politics. Available at SSRN 4395173 .
Glaeser, E. L. and D. C. Mare (2001). Cities and skills. Journal of Labor Economics 19 (2), 316–342.
Goldsmith-Pinkham, P., I. Sorkin, and H. Swift (2020). Bartik instruments: What, when, why, and how.
American Economic Review 110 (8), 2586–2624.
Graetz, G. (2020). Labor demand in the past, present and future. CEP Discussion Paper No 1683 .
Graetz, G. and G. Michaels (2018). Robots at work. Review of Economics and Statistics 100 (5), 753–768.
Guiso, L., H. Herrera, M. Morelli, and T. Sonno (2019). Global crises and populism: the role of Eurozone
institutions. Economic Policy 34 (97), 95–139.
Guiso, L., H. Herrera, M. Morelli, T. Sonno, et al. (2017). Demand and Supply of Populism. Centre for
Economic Policy Research London, UK.
Guiso, L., H. Herrera, M. Morelli, T. Sonno, et al. (2020). Economic insecurity and the demand of
populism in Europe. Einaudi Institute for Economics and Finance.
42
Guriev, S., N. Melnikov, and E. Zhuravskaya (2021). 3G internet and confidence in government. The
Quarterly Journal of Economics 136 (4), 2533–2613.
Guriev, S. and E. Papaioannou (2022). The political economy of populism. Journal of Economic Literature 60 (3), 753–832.
Halla, M., A. F. Wagner, and J. Zweimüller (2017). Immigration and voting for the far right. Journal of
the European Economic Association 15 (6), 1341–1385.
Heins, G. (2023). Endogenous vertical differentiation, variety, and the unequal gains from trade. Review
of Economics and Statistics 105 (4), 910–930.
Helpman, E., O. Itskhoki, M.-A. Muendler, and S. J. Redding (2016, 06). Trade and Inequality: From
Theory to Estimation. The Review of Economic Studies 84 (1), 357–405.
Inglehart, R. F. and P. Norris (2016). Trump, Brexit, and the rise of populism: Economic have-nots and
cultural backlash.
Kambourov, G. and I. Manovskii (2009). Occupational mobility and wage inequality. The Review of
Economic Studies 76 (2), 731–759.
Kambourov, G. and I. Manovskii (2013). A cautionary note on using (march) current population survey
and panel study of income dynamics data to study worker mobility. Macroeconomic Dynamics 17 (1),
172–194.
Lewis, J. B., K. Poole, H. Rosenthal, A. Boche, A. Rudkin, and L. Sonnet (2023). Voteview: Congressional
roll-call votes database. See https://voteview. com/(accessed 30 May 2023).
Malgouyres, C. (2017). Trade shocks and far-right voting: Evidence from French presidential elections.
Robert Schuman Centre for Advanced Studies Research paper no. RSCAS 21.
Manacorda, M., G. Tabellini, and A. Tesei (2022). Mobile internet and the rise of political tribalism in
Europe. Technical report, London School of Economics and Political Science, LSE Library.
Margalit, Y. (2019). Political responses to economic shocks. Annual Review of Political Science 22 (1),
277–295.
Marinescu, I. and R. Rathelot (2018, July). Mismatch unemployment and the geography of job search.
American Economic Journal: Macroeconomics 10 (3), 42–70.
Meltzer, A. H. and S. F. Richard (1981). A rational theory of the size of government. Journal of Political
Economy 89 (5), 914–927.
Mian, A. and A. Sufi (2014). What explains the 2007–2009 drop in employment? Econometrica 82 (6),
2197–2223.
Mocan, N. and C. Raschke (2016). Economic well-being and anti-semitic, xenophobic, and racist attitudes
in Germany. European Journal of Law and Economics 41 (1), 1–63.
Moscarini, G. and K. Thomsson (2007). Occupational and job mobility in the US. The Scandinavian
Journal of Economics 109 (4), 807–836.
Moscarini, G. and F. G. Vella (2008). Occupational mobility and the business cycle. Technical report,
National Bureau of Economic Research.
Newell, R. G. and J. Siikamäki (2015). Individual time preferences and energy efficiency. American
Economic Review 105 (5), 196–200.
Nokken, T. P. and K. T. Poole (2004). Congressional party defection in American history. Legislative
Studies Quarterly 29 (4), 545–568.
Nosnik, I. P., P. Friedmann, H. M. Nagler, and C. Z. Dinlenc (2010). Resume fraud: Unverifiable
publications of urology training program applicants. The Journal of Urology 183 (4), 1520–1523.
Noury, A. and G. Roland (2020). Identity politics and populism in Europe. Annual Review of Political
Science 23 (1), 421–439.
Panunzi, F., N. Pavoni, and G. Tabellini (2020). Economic shocks and populism. Technical report.
43
Paolillo, A., F. Colella, N. Nosengo, F. Schiano, W. Stewart, D. Zambrano, I. Chappuis, R. Lalive,
and D. Floreano (2022). How to compete with robots by assessing job automation risks and resilient
alternatives. Science Robotics 7 (65), eabg5561.
Patnaik, A., J. Venator, M. Wiswall, and B. Zafar (2022). The role of heterogeneous risk preferences,
discount rates, and earnings expectations in college major choice. Journal of Econometrics 231 (1),
98–122.
Piketty, T. (2014). Capital in the Twenty-First Century. Harvard University Press.
Piketty, T. and G. Zucman (2014, 05). Capital is Back: Wealth-Income Ratios in Rich Countries 1700-
2010. The Quarterly Journal of Economics 129 (3), 1255–1310.
Putnam, R. (2015). Our Kids: The American Dream in Crisis. Simon & Schuster.
Rodrik, D. (2018). Populism and the economics of globalization. Journal of International Business
Policy 1 (1-2), 12–33.
Samuel, L. R. (2012). The American dream: A cultural history. Syracuse University Press.
Schubert, G., A. Stansbury, and B. Taska (2021). Employer concentration and outside options. Available
at SSRN 3599454 .
Serani, D. (2016). Explaining vote for populist parties: the impact of the political trust, the economic
and the political context. University of Pompeu Fabra working paper .
Solon, G. (1999). Intergenerational mobility in the labor market. In Handbook of Labor Economics,
Volume 3, pp. 1761–1800. Elsevier.
Topel, R. H. and M. P. Ward (1992). Job mobility and the careers of young men. The Quarterly Journal
of Economics 107 (2), 439–479.
Vom Lehn, C., C. Ellsworth, and Z. Kroff (2022). Reconciling occupational mobility in the current
population survey. Journal of Labor Economics 40 (4), 1005–1051.
Xu, M. (2018). Understanding the decline in occupational mobility. Technical report, Working Paper.
44
A ONLINE APPENDIX: Additional Figures & Tables (Not for Publication)
A.1 BGT Data Characteristics
FigureA.1 plots the share of occupations in different industries listed in the Burning Glass Technologies
sample vs. the shares reported by the Bureau of Labor Statistics (BLS).
Figure A.2 maps the shares of resumes in the BGT data by county.
Figure A.1: Comparison of BGT and Bureau of Labor Statistics Share of Occupations
Notes. Figure represents the share of occupations in various industries in the Burning Glass Technologies sample vs. the
shares reported by the Bureau of Labor Statistics (BLS) in 2016.
Figure A.2: Density of Resumes in BGT Data per County
Notes. The figure represents the share of resumes in the BGT dataset per 100 people in each county. County population
data is obtained from American Community Survey 5-yearly estimates. Only candidates with a job between 2004 and 2016
are plotted in the figure.
A1
A.2 Occupational Transitions
Figures A.3 and A.4 replicate Figure 3 of the paper for construction and installation occupations, respectively, based on the BGT sample.
Figure A.3: Occupational Transitions From Construction Occupations, 2004 vs 2016
Notes. The figures show the normalized probabilities of transitioning from construction occupations to various other occupation groups. The left figure demonstrates the
transitions for 2004 and the right figure demonstrates the transitions for 2016. The size of each bubble indicates the likelihood of transition to the labeled occupation
group. Normalized occupational transition probabilities are transition probabilities divided by the probability of occupational transition for the whole sample in that
year.
A2
Figure A.4: Occupational Transitions From Installation Occupations, 2004 vs 2016
Notes. The figures show the normalized probabilities of transitioning from installation occupations to various other occupation groups. The left figure demonstrates the
transitions for the year 2004 and the right figure demonstrates the transitions for 2016. Size of each bubble indicates the likelihood of transition to the labeled occupation
group. Normalized occupational transition probabilities are transition probabilities divided by the probability of occupational transition for the whole sample in that
year.
A3
A.3 Comparison Between BGT and CPS Data
Figure A.5 shows the likelihood of transitioning into a job with higher, lower, or similar wage to the previous job, conditional on moving
between detailed occupations, for 1981-2020 using CPS data. Figure A.6 shows occupational mobility among male workers age 20-64 measured
biannually using the CPS Employee Tenure and Occupational Mobility Supplement. Figure A.7 demonstrates the transitions to occupations
at higher/similar/lower wage levels using CPS data.
Figure A.5: Probabilities of Transition Out of Occupations in the CPS
(a) Detailed occ. mobility (b) Broad occ. group mobility
Notes. Figure shows 1-year occupational mobility measured in CPS Employee Tenure and Occupational Mobility Supplement data among male workers age 20-64 who
are employed at the time of the survey for both transitions out of detailed harmonized 1990 Census occupations (left panel), as well as between broad occupation groups
(right panel). The mobility rates are averaged by broad occupation groups for better visibility in the graph. The bottom axis averages mobility rates in 2000, 2002, and
2004, and the vertical axis for 2010, 2012, and 2014. The red line in each graph indicates the 45 degree line corresponding to no change in mobility between time periods.
A4
Figure A.6: Occupational mobility in Current Population Survey data
Notes. Figure shows 1-year occupational mobility measured biannually in the CPS Employee Tenure and Occupational
Mobility Supplement among male workers age 20-64 who are employed at the time of the survey for both transitions out of
detailed harmonized 1990 Census occupations, as well as between broad occupation groups.
A.4 Career Values and Automatability of Occupations
Figure A.8 displays the correlation between the change in career values of occupations between 2000 and
2016 calculated based on the BGT data and the automatability score of the occupation from Frey and
Osborne (2017).
A.5 Voting Intentions
Using data from the Congressional Cooperative Election Survey, Figure A.9 breaks down the stated voting
intention for Trump in the 2016 Presidential Race. The voting intentions are highest among the whites
and non-Hispanics.
A5
Figure A.7: Transitions to Different Types of Occupations in CPS data, by Time Period
Notes. Figure shows the probability in % that hourly wages increase by more than 10% (left), decline by more than 10%
(right), or experience a smaller absolute change (middle) at a 1-year horizon in longitudinally linked CPS March Surveys,
conditional on the worker experiencing a change in detailed harmonized 1990 Census occupation. The sample consists of
male workers age 20-64 who are employed at the time of the survey.
Figure A.8: Career Values vs Frey and Osborne Automatability Score, by 2-digit SOC
Career Value Changes 2001-2016 by Automatability
Automatability: Probability of computerisation by Frey & Osborne (2017) – prediction
score for “Can the tasks of this job be suciently specified, conditional on the availability
of big data, to be performed by state of the art computer-controlled equipment”:
20/72
Career Value Changes 2001-2016 by Automatability
Automatability: Probability of computerisation by Frey & Osborne (2017) – prediction score for
“Can the tasks of this job be suciently specified, conditional on the availability of big data, to
be performed by state of the art computer-controlled equipment”:
Petrova et al. (2020) Automation, Career Values, and Politics 26/10/2021 21 / 59
Notes. The figure depicts the correlation between the change in career values of occupations between 2000 and 2016
and the automatability score of the occupation based on Frey and Osborne (2017), which calculates the susceptibility of
occupational groups to automation (computerization) based on machine learning tools. Career values are derived based on
the methodology described in Section 3.1.
A6
Figure A.9: Voting Intentions by Race, 2016 Presidential Elections
Notes. The source of data is the Congressional Cooperative Election Survey (2016). Numbers indicate the share of individuals
by race out of those intending to vote for Trump.
A7
Additional Tables
A.6 Summary statistics
Table A.1: Summary Statistics, 2004-2016
Variable Obs Mean Std. Dev. Min Max P25 P75
Local Market Career Values, 04-08 741 -.088 1.091 -4.179 3.892 -.786 .652
Local Market Career Values, 04-16 741 -.103 1.516 -4.943 4.273 -1.115 .857
Local Market Career Values, 08-16 741 -.015 1.254 -3.161 3.625 -.879 .821
Automation Shock US, 04-08 722 .583 .609 .053 5.286 .218 .733
Automation Shock US, 04-16 722 1.419 1.132 .195 9.033 .659 1.837
Automation Shock US, 08-16 722 .836 .542 .136 3.747 .438 1.085
Automation Shock Denmark, 04-16 722 2.228 1.897 -1.291 13.901 .973 3.077
Automation Shock Sweden, 04-16 722 1.475 .736 -1.029 4.967 .972 1.773
Automation Shock Italy, 04-16 722 .569 .781 -6.180 4.981 .290 .871
Automation Shock France, 04-16 722 .311 .315 -3.224 1.038 .230 .437
Automation Shock Finland, 04-16 722 .374 1.722 -5.170 20.617 -.205 .511
Automation Shock Germany, 04-16 722 .846 1.298 -2.243 13.658 .315 6.187
Automation Shock Spain, 04-16 722 .790 .393 .191 2.230 .466 1.018
Automation Shock UK, 04-16 722 .465 .216 .111 1.142 .288 .592
Female Share, 1990 722 .511 .009 .477 .544 .505 .517
Hispanic Share, 1990 722 .057 .118 .001 .939 .006 .046
White Share, 1990 722 .871 .12 .371 .993 .801 .966
Black Share, 1990 722 .076 .116 0 .615 .004 .086
Asian Share, 1990 722 .022 .049 0 .591 .003 .016
High School Degree, 1990 722 .619 .081 .349 .821 .574 .675
College Degree, 1990 722 .101 .031 .039 .233 .078 .118
Master Degree, 1990 722 .032 .012 .014 .106 .024 .038
Above 65 years, 1990 722 .136 .028 .059 .301 .118 .154
Log of population, 1990 722 11.468 1.574 7.155 16.479 10.487 12.478
Manufacturing Share, 1990 722 .224 .104 .043 .554 .139 .292
Female Share in Manufacturing, 1990 722 .077 .047 .014 .245 .041 .102
Light Manufacturing Share, 1990 722 .054 .05 .003 .433 .02 .063
Automation Shock China, 90-07 722 2.947 2.855 -.112 27.958 .91 4.144
Vote share of McCain, 2008 726 56.873 13.048 19.798 90.076 48.007 66.271
Vote share of Romney, 2012 726 60.015 13.986 20.494 91.281 50.307 70.585
Vote share of Trump, 2016 726 65.178 14.922 20.093 95.051 54.936 76.961
18+ school attendance in 2000 722 .067 .025 .033 .249 .05 .075
College rate, 2000 722 .123 .039 .056 .301 .095 .142
Intergenerational mobility, Chetty et al., 718 .025 .435 -1.38 1.431 -.29 .28
causal impact of neighborhoods
Empl. Change, High School Degree, 2000-2015 722 -.241 3.064 -8.675 8.113 -2.538 1.816
Empl. Change, College Degree, 2000-2015 722 -2.658 2.328 -9.8 4.615 -4.171 -1.285
Empl. Change, Some College Degree, 2000-2015 722 -2.317 2.301 -9.476 4.425 -3.875 -.784
Empl. Change, Master Degree, 2000-2015 722 -3.594 3.502 -16.916 4.977 -5.755 -1.26
Wage Change, High School Degree, 2000-2015 722 .083 .093 -.131 .416 .02 .126
Wage Change, College Degree, 2000-2015 722 .083 .095 -.121 .441 .021 .129
Wage Change, Some College Degree, 2000-2015 722 .117 .092 -.132 .439 .056 .159
Wage Change, Master Degree, 2000-2015 722 .106 .109 -.231 .556 .034 .171
A8
A.7 Career Value Decomposition and Occupational Changes: Own vs. Other Occupations
Table A.2 decomposes the changes in career value into occupational mobility and wage change expectations for an employee’s current (own) occupation and other occupations. Higher automation exposure
lowers lifetime earnings from both one’s own career and other occupations, however, the magnitude of
the decline is much larger for the other, future occupations.
Table A.3 summarizes the effect of robotization on the change in the probability of moving away from
an occupation (conditional on moving away from an occupation), mapped to classifications of occupations
based on whether the occupation is considered manual (column 1) vs. cognitive (column 2) and routine
(column 3) vs. non-routine (column 4), as well as the combinations of the two dimensions (columns
5-8) according to the definitions from Autor et al. (2003). The change in the conditional probability
of transitioning away from manufacturing occupations is also provided as a reference in column 9. The
table indicates a positive change in the likelihood of moving away from manual job categories (column
4) for both routine (column 5) and non-routine manual (column 6) occupations in regions that were
more exposed to robotization. We observe a positive and significant effect of similar magnitude for the
manufacturing occupations (column 9). The effect on cognitive occupations is not significant.
Table A.4 decomposes the effects of exposure to robots on the likelihood of changing occupations
(column 1), changing employers (column 2), changing employers while keeping one’s occupation (column
3), and changing occupation while staying with one’s firm (column 4). The table then breaks down the
likelihood of occupational movements to and from management (columns 5 and 6), engineering (columns
7 and 8), construction (columns 9 and 10), maintenance (columns 11 and 12), and production occupations
(columns 13 and 14). The estimates in this table suffer from weak identification and survivorship bias.
However, focusing on the two columns with higher F stats, we can conclude that higher robot exposure
in a region results in higher horizontal mobility—changing employer while keeping a job within the same
occupation family (column 3). In addition, higher exposure to robots implies a higher movement from
production occupations (column 13).
A9
Table A.2: Exposure to Robots and Career Value (CV) Changes (Own vs. Other Occupations). IV Lasso
∆ CV [Own Occupation) ∆ CV (Other Occupation)
Total ∆w ∆ψ ∆ψ × ∆w Total ∆w ∆ψ ∆ψ × ∆w
(1) (2) (3) (4) (5) (6) (7) (8)
U.S. Robot Exposure ’04-’16 -0.043*** -0.060*** 0.022** -0.004*** -0.292*** -0.144*** -0.138*** -0.009
[0.010] [0.013] [0.010] [0.001] [0.035] [0.032] [0.022] [0.007]
Mean of D.V. -0.319 0.244 -0.540 -0.024 0.076 0.828 -0.600 -0.152
F-stat. 23.135 90.346 12.527 24.101 66.811 44.452 25.712 4.907
Census Division FE Yes Yes Yes Yes Yes Yes Yes Yes
A&R Controls Yes Yes Yes Yes Yes Yes Yes Yes
Observations 722 722 722 722 722 722 722 722
Note. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in parentheses
and clustered by state. All columns include census division dummies as well as controls for demographic characteristics
of commuting zones in 1990 identical to those in Acemoglu and Restrepo (2020) (log population; the share of females;
the share of population above 65 years; the shares of population with no college, some college, college and professional
degrees, and masters and doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment
in manufacturing; the share of female employment in manufacturing; and the share of employment in light manufacturing,
namely, textile industry and the paper, publishing, and printing industry). U.S. Robot Exposure is instrumented using robot
exposure in Denmark, Italy, Sweden, Finland, France, Germany, Spain and the UK in the same time periods. Observations
are weighted by commuting zone population.
Table A.3: Exposure to Robots and Probability of Transitions From Occupations (IV)
∆ Conditional Probability Occupational Transition From
Manual Manual Cognitive Cognitive
Manual Cognitive Non-Routine Routine Routine Non-Routine Routine Non-Routine Manufacturing
(1) (2) (3) (4) (5) (6) (7) (8) (9)
U.S. Robot Exposure ’04â’16 0.01691** 0.00087 0.01421 0.01157 0.01601* 0.01871*** 0.00658 0.00860 0.01249*
(0.00752) (0.01080) (0.01154) (0.00806) (0.00886) (0.00607) (0.01138) (0.00969) (0.00649)
Mean of D.V. -0.013 0.008 0.022 -0.029 0.011 -0.024 0.024 -0.014 -0.034
F-stat. 1344.537 1344.537 1344.537 1344.537 1344.537 1344.537 1344.537 1344.537 1344.537
Census Division FE Yes Yes Yes Yes Yes Yes Yes Yes Yes
A&R Controls Yes Yes Yes Yes Yes Yes Yes Yes Yes
Observations 722 722 722 722 722 722 722 722 722
Notes. *** p<0.01, ** p<0.05, * p<0.1. Table reports the likelihood of transitions among the individuals who are changing
occupations. Robust standard errors are in parentheses and clustered by state. All columns include census division dummies
as well as controls for demographic characteristics of commuting zones in 1990 identical to those in Acemoglu and Restrepo
(2020) (log population; the share of females; the share of population above 65 years; the shares of population with no college,
some college, college and professional degrees, and masters and doctoral degrees; and the shares of Whites, Blacks, Hispanics,
and Asians). U.S. Robot Exposure is instrumented using robot exposure in Denmark, Italy, Sweden, Finland, France.
A10
Table A.4: Exposure to Robots and Job Transitions (Change in Occupations within and between Firms. IV Lasso)
(1) (2) (3) (4) (5) (6) (7)
Change occupation Change firm Change firm Change occupation Change occupation Change occupation Change occupation
within occupation within firm from management to management from engineering
U.S. Robot Exposure ’04-’16 -0.002** 0.002*** 0.003*** 0.000 0.001 0.002** 0.007*
[0.001] [0.001] [0.001] [0.000] [0.001] [0.001] [0.004]
Mean of D.V. .67 1.08 .17 .04 .65 .24 .86
F-stat. 3.99 5.19 14.35 .2 .06 4.13 1.97
Observations 722 722 722 722 721 722 696
(8) (9) (10) (11) (12) (13) (14)
Change occupation Change occupation Change occupation Change occupation Change occupation Change occupation Change occupation
to engineering from construction to construction from maintenance to maintenance from production to production
U.S. Robot Exposure ’04-’16 0.001*** -0.011*** -0.000 0.001 0.000 0.014*** -0.000
[0.000] [0.004] [0.000] [0.003] [0.000] [0.004] [0.000]
Mean of D.V. .03 1.02 .02 .93 .03 .98 .04
F-stat. 4.43 4.9 .27 .03 .4 13.35 .44
Observations 722 683 722 697 722 685 722
A&R Controls Yes Yes Yes Yes Yes Yes Yes
Census Division FE Yes Yes Yes Yes Yes Yes Yes
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets and clustered by state. All columns include census
division dummies as well as controls for demographic characteristics of commuting zones in 1990 identical to those in Acemoglu and Restrepo (2020) (log population;
the share of females; the share of population above 65 years; the shares of population with no college, some college, college and professional degrees, and masters and
doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment in manufacturing; the share of female employment in manufacturing;
and the share of employment in light manufacturing, namely, textile industry and the paper, publishing, and printing industry). U.S. Robot Exposure is instrumented
using robot exposure in Denmark, Italy, Sweden, Finland, France, Germany, Spain and the UK in the same time periods. Observations are weighted by commuting zone
population.
A11
A.8 Heterogeneity of Automation Effects with Respect to Upward Mobility
In this section, we repeat the estimation of equation (8) using alternative proxies for opportunities for
upward mobility, more specifically, the average level of education (Table A.5) and local intergenerational
mobility (Table A.6). The measure of the latter are the neighborhood intergenerational mobility effects
from Chetty et al. (2016) The results are qualitatively similar to the results in Table 8.
Table A.5: Automation, Career Values, and Education
∆ Career Values 2000-2016 ∆ Wage 2000-2015 ∆ Employment 2000-2015
All Below College College or Higher
(1) (2) (3) (4) (5)
EU Instrument ’04-’16 -0.086* -0.200** -0.051 -0.003* -0.020
[0.049] [0.092] [0.035] [0.002] [0.060]
College Rate ’00 7.803 -4.304 12.754* 0.853** -3.479
[6.289] [9.088] [6.778] [0.340] [7.137]
EU Instrument ’04-’16 x College Rate ’00 0.170 0.836 -0.059 0.006 -0.184
[0.286] [0.513] [0.221] [0.010] [0.378]
Mean of D.V. -0.139 -0.232 -0.262 0.135 0.036
R-squared 0.491 0.242 0.447 0.545 0.746
Census Division FE Yes Yes Yes Yes Yes
A&R Controls Yes Yes Yes Yes Yes
Observations 722 722 722 722 722
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets
and clustered by state. All columns include census division dummies as well as controls for demographic characteristics
of commuting zones in 1990 identical to those in Acemoglu and Restrepo (2020) (log population; the share of females;
the share of population above 65 years; the shares of population with no college, some college, college and professional
degrees, and masters and doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment
in manufacturing; the share of female employment in manufacturing; and the share of employment in light manufacturing,
namely, textile industry and the paper, publishing, and printing industry). The EU instrument is the average robot exposure
in Denmark, France, Finland, UK and Germany during 2004-2016. Observations are weighted by commuting zone population.
A.9 Creative Destruction in Tradable/Non-Tradable Sectors
In this subsection, we ask what happens to the patterns of business creation and destruction in different
sectors. If demand for services/products are shrinking in a region, we may also anticipate the number
of establishments in an area to go down. We follow Mian and Sufi (2014) (MS from here on), who
use 4-digit NAICS codes in defining tradable vs. non-tradable establishments, to estimate this effect.
Accordingly, an establishment is classified as tradable “if it has imports plus exports equal to at least
$10,000 per worker, or if total exports plus imports for the NAICS 4-digit industry exceeds $500M.” For
example, retail and restaurant industries are non-tradable, while manufacturing, mining, or agriculture –
i.e. industries that appear in global trade data—are tradable.24 These results are reported in Table A.7.
Although we cannot separate business entries from business exits, we see negative signs in the overall
number of establishments in places with higher exposure to robots, while the effects are not statistically
significant. Both trade-able and non-tradable establishments have fewer number of employees (columns
4 and 5), with the numbers of employees in non-tradable sectors more negatively impacted. We see a
higher average number of employees in the construction sector, which, together with the negative signs
for the number of businesses, may imply a trend toward monopolization. Overall, these results suggest
24We also use a more restricted version of non-tradable industries that includes only grocery retail stores and restaurants.
A12
Table A.6: Automation, Career Values, and Intergenerational Mobility
∆ Career Values 2000-2016 ∆ Wage 2000-2015 ∆ Employment 2000-2015
All Below College College or Higher
(1) (2) (3) (4) (5)
EU Instrument ’04-’16 -0.155*** -0.175** -0.165*** -0.001 -0.032
[0.052] [0.068] [0.056] [0.002] [0.043]
Intergenerational Mobility 10.048** 6.435 12.993*** 0.819*** 37.634***
[4.287] [5.793] [4.694] [0.177] [4.571]
EU Instrument ’04-’16 x Intergenerational Mobility 1.459* 1.486 1.647* -0.016 -0.258
[0.862] [1.197] [0.924] [0.030] [0.735]
Mean of D.V. -0.162 -0.246 -0.286 0.133 -0.013
R-squared 0.418 0.179 0.384 0.556 0.759
Census Division FE Yes Yes Yes Yes Yes
A&R Controls Yes Yes Yes Yes Yes
Observations 712 712 712 712 712
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets
and clustered by state. All columns include census division dummies as well as controls for demographic characteristics
of commuting zones in 1990 identical to those in Acemoglu and Restrepo (2020) (log population; the share of females;
the share of population above 65 years; the shares of population with no college, some college, college and professional
degrees, and masters and doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment
in manufacturing; the share of female employment in manufacturing; and the share of employment in light manufacturing,
namely, textile industry and the paper, publishing, and printing industry). The EU instrument is the average robot exposure
in Denmark, France, Finland, UK and Germany during 2004-2016. Intergenerational mobility measures are the neighborhood
mobility measures from (Chetty et al., 2016). Observations are weighted by commuting zone population.
that, there may be some negative effects of automation on local business growth, consistent with declining
demand for goods and services.
A.10 Automation, Career Value, Employment, and Wages by Industry
Table A.8 regresses the change in career values in broad 2-digit NAICS industry sectors (and selected
sub-sectors) on exposure to robots, where columns (3) to (15) correspond to the manufacturing related
industries. We find that the change in career value is not only confined to manufacturing jobs. In regions
more exposed to automation, career values also declined in utilities (16), research (18), and services (19).
Put differently, there is a broad decline in the future labor market conditions. The magnitude of the
decline in these categories is comparable to that in a number of manufacturing categories.
Table A.9 regresses the change in employment in a CZ for all, private, and public jobs, and then
specifically for manufacturing and service occupations. The takeaway from the table is that, increasing
robotization goes hand in hand with a decline in employment in all five categories (Panel A), and the
overall effect is driven by the declines in the manufacturing intensive CZs (Panel B).
Table A.10 regresses the change in employment in all sectors classified according to the 2-digit SOC
classification. We again find that the change in employment is not only confined to manufacturing jobs
(Panel A, columns (5)-(7)). In regions that were more exposed to automation, employment in a number
of occupation categories decline, including retail (Panel B, columns (1) and (2)), transportation and
warehousing (Panel B, column (3)), as well as jobs that are of more white collar nature (professional,
scientific, technical services (Panel B column (8)). Put differently, there is a broader job decline in the
labor market conditions in the regions that are more exposed to automation beyond manufacturing jobs.
We report the aggregate wage changes in regions exposed to automation in Table A.11. Accordingly,
with increasing automation exposure, wages in all reported sectors decline. On average, the hourly wages
A13
Table A.7: Automation and Establishments, by Sector. IV Lasso
No. of Establishments No. of Employees Per Establishment
Tradable Non-tradable Construction Tradable Non-tradable Construction
(1) (2) (3) (4) (5) (6)
U.S. Robot Exposure ’04-’16 0.000 0.010 -0.001 -0.010** -0.022*** 0.017***
[0.003] [0.008] [0.003] [0.005] [0.003] [0.004]
Mean of D.V. -0.082 -0.013 -0.119 0.128 -0.078 0.037
F-stat. 0.007 2.212 0.515 18.018 51.354 15.233
Census Division FE Yes Yes Yes Yes Yes Yes
A&R Controls Yes Yes Yes Yes Yes Yes
Observations 716 712 722 721 722 722
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets
and clustered by state. All columns include census division dummies as well as controls for demographic characteristics
of commuting zones in 1990 identical to those in Acemoglu and Restrepo (2020) (log population; the share of females;
the share of population above 65 years; the shares of population with no college, some college, college and professional
degrees, and masters and doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment
in manufacturing; the share of female employment in manufacturing; and the share of employment in light manufacturing,
namely, textile industry and the paper, publishing, and printing industry). U.S. Robot Exposure is instrumented using
robot exposure in Denmark, Italy, Sweden, Finland, France, Germany, Spain and the UK in the same time periods.Sectoral
classification is based on 4-digit NAICS. Observations are weighted by commuting zone population.
decline by $0.01 with 1 additional robot per 1000 workers (column (1)). A breakdown of the CZs into
high manufacturing (columns (4)-(6)) and low manufacturing (columns (7)-(9)) areas demonstrate that
most of the wage declines are observed in the manufacturing intensive zones.
Table A.12 reports how the average hourly wage in different industries changes with automation
exposure between 2004 and 2016 and paints a similar picture to Table A.11 in that automation exposure
results in wage declines in a number of industries, even in the ones that are not directly impacted by
industrial robots.
A14
Table A.8: Exposure to Robots vs Sector Career Values, by Industry Groups. IV Lasso
(1) (2) (3) (4) (5) (6) (7) (8) (9)
Agriculture Mining Food Textiles Furniture Paper Petrochemicals Mineral Metal Basic
U.S. Robot Exposure ’04-’16 -0.397*** -0.290*** -0.453*** -0.479*** -0.229*** -0.367*** -0.364*** -0.227*** -0.448***
[0.051] [0.090] [0.066] [0.058] [0.043] [0.046] [0.060] [0.086] [0.040]
Mean of D.V. -0.278 0.422 -1.019 -1.01 -0.023 -1.077 -0.202 -0.368 0.306
F-stat. 82.03 31.31 111.71 72.42 25.71 83.55 62.8 8 56.45
(10) (11) (12) (13) (14) (15) (16) (17) (18) (19)
Metal Machinery Metal Products Electronics Automotive Vehicles (Other) Manufacturing (Other) Utilities Construction Research Services
U.S. Robot Exposure ’04-’16 -0.166*** -0.816*** -0.342*** -0.068 -0.837*** -0.111 -0.388*** -0.406*** -0.490*** -0.366***
[0.045] [0.087] [0.063] [0.068] [0.152] [0.122] [0.031] [0.042] [0.051] [0.039]
Mean of D.V. -0.715 0.217 0.357 -0.638 2.057 -0.893 0.154 -0.358 -0.345 -0.192
F-stat. 23.57 90.82 38.83 4.26 29.89 3.07 99.37 134.7 120.88 90.55
Census Division FE Yes Yes Yes Yes Yes Yes Yes Yes Yes Yes
A&R Controls Yes Yes Yes Yes Yes Yes Yes Yes Yes Yes
Observations 722 526 722 722 440 681 722 722 722 722
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets and clustered by state. All columns include census
division dummies as well as controls for demographic characteristics of commuting zones in 1990 identical to those in Acemoglu and Restrepo (2020) (log population;
the share of females; the share of population above 65 years; the shares of population with no college, some college, college and professional degrees, and masters and
doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment in manufacturing; the share of female employment in manufacturing; and
the share of employment in light manufacturing, namely, textile industry and the paper, publishing, and printing industry). U.S. Robot Exposure is instrumented using
robot exposure in Denmark, Italy, Sweden, Finland, France, Germany, Spain and the UK in the same time periods. Industrial classification is based on 2-digit NAICS.
Observations are weighted by commuting zone population.
A15
Table A.9: Employment by Industry and Automation Shocks, 2004-2016. IV Lasso
Employment and Automation Shocks, 2004-2016. IV Lasso
Panel A. Full Sample Change in employment 2000-2015 (log), ACS
Aggregate Private Public Manufacturing Services
(1) (2) (3) (4) (5)
U.S. Robot Exposure ’04-’16 -0.009** -0.010** -0.004 -0.001 -0.009**
[0.004] [0.004] [0.004] [0.006] [0.004]
Mean of D.V. 0.073 0.107 0.044 -0.152 0.162
F-stat. 8.53 9.370 1.07 0.004 6.99
Census Division FE Yes Yes Yes Yes Yes
A& R Controls Yes Yes Yes Yes Yes
Observations 722 722 722 722 722
Panel B. Employment in Change in employment 2000-2015 (log), ACS
manufacturing intensive areas % manufacturing above the median
Aggregate Private Public Manufacturing Services
(6) (7) (8) (9) (10)
U.S. Robot Exposure ’04-’16 -0.008*** -0.009*** -0.007* -0.011* -0.006**
[0.002] [0.003] [0.004] [0.006] [0.002]
Mean of D.V. 0.037 0.056 0.032 -0.259 0.155
F-stat. 9.29 9.69 4.21 4.18 4.69
Census Division FE Yes Yes Yes Yes Yes
A& R Controls Yes Yes Yes Yes Yes
Observations 722 722 722 722 722
Panel C. Employment in other areas Change in employment 2000-2015 (log), ACS
% manufacturing below the median
Aggregate Private Public Manufacturing Services
(11) (12) (13) (14) (15)
U.S. Robot Exposure ’04-’16 0.082 0.075 0.318 0.375 0.056
[0.137] [0.144] [0.198] [0.373] [0.136]
Mean of D.V. 0.108 0.158 0.056 -0.044 0.168
F-stat. 0.78 0.64 7.60 4.56 0.31
Census Division FE Yes Yes Yes Yes Yes
A& R Controls Yes Yes Yes Yes Yes
Observations 722 722 722 722 722
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets
and clustered by state.All columns include census division dummies as well as controls for demographic characteristics
of commuting zones in 1990 identical to those in Acemoglu and Restrepo (2020) (log population; the share of females;
the share of population above 65 years; the shares of population with no college, some college, college and professional
degrees, and masters and doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment
in manufacturing; the share of female employment in manufacturing; and the share of employment in light manufacturing,
namely, textile industry and the paper, publishing, and printing industry). U.S. Robot Exposure is instrumented using robot
exposure in Denmark, Italy, Sweden, Finland, France, Germany, Spain and the UK in the same time periods. Change in
logarithms for employment is computed as the difference of logarithms in numbers of those employed in a given sector in
between 2015 and 2000. Observations are weighted by commuting zone population.
A16
Table A.10: Exposure to Robots vs Sector Employment, by Industry Groups
Change in Employment by Industry (log), ACS, 2000-2015
(1) (2) (3) (4) (5) (6) (7) (8)
Agriculture, Forestry, Mining, Quarring, Utilities Construction Manufacturing Wholesale
Fishing & Hunting Oil & Gas Extraction (NAICS 31) (NAICS 32) (NAICS 33) Trade
Panel A
U.S. Robot Exposure ’04-’16 0.022* -0.001 0.002 -0.026*** -0.013 0.028*** -0.005 -0.022***
[0.013] [0.023] [0.011] [0.006] [0.008] [0.011] [0.006] [0.006]
Mean of D.V. 0.136 0.459 0.072 0.046 -0.078 -0.187 -0.112 -0.151
F-stat. 9.366 0.038 0.206 27.462 1.728 14.338 0.015
Retail Trade Transportation & Warehousing Information Finance Real Estate & Rental Professional, Scientific,
(NAICS 44) (NAICS 45) (NAICS 48) (NAICS 49) & Insurance & Leasing & Technical Serv.
Panel B
U.S. Robot Exposure ’04-’16 -0.016*** -0.015* -0.011* -0.009 -0.005 -0.004 -0.005 -0.023***
[0.004] [0.009] [0.007] [0.016] [0.009] [0.006] [0.006] [0.008]
Mean of D.V. 0.066 0.066 0.08 0.135 -0.159 0.07 0.097 0.276
F-stat. 9.568 14.3 4.063 1.559 0.483 0.907 0.375
Mgmt of Companies Admin. & Support Educational Serv. Health Care Arts, Entertainment, Accommodation Other Serv.
& Enterprises & Waste Mgmt and Social Assist. & Recreation Food Serv. (except Public Admin.)
& Remediation Serv.
Panel C
U.S. Robot Exposure ’04-’16 -0.004 0.001 -0.007 -0.002 0.010* -0.004 -0.006
[0.010] [0.006] [0.006] [0.005] [0.006] [0.005] [0.004]
Mean of D.V. 0.338 0.426 0.107 0.339 0.331 0.236 -0.013
F-stat. 0.908 16.816 0.04 0.35 1.115 0.394 0.737
Census Division FE Yes Yes Yes Yes Yes Yes Yes Yes
A& R Controls Yes Yes Yes Yes Yes Yes Yes Yes
Observations 722 722 722 722 722 722 722 722
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets and clustered by state. All columns include census
division dummies as well as controls for demographic characteristics of commuting zones in 1990 identical to those in Acemoglu and Restrepo (2020) (log population;
the share of females; the share of population above 65 years; the shares of population with no college, some college, college and professional degrees, and masters and
doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment in manufacturing; the share of female employment in manufacturing;
and the share of employment in light manufacturing, namely, textile industry and the paper, publishing, and printing industry). U.S. Robot Exposure is instrumented
using robot exposure in Denmark, Italy, Sweden, Finland, France, Germany, Spain and the UK in the same time periods. Industrial classification is based on 2-digit
NAICS. Change in logarithms of employment is computed as the difference in logarithms for the numbers of those employed in a given industry between 2015 and 2000.
Observations are weighted by commuting zone population.
A17
Table A.11: Exposure to Automation and Wages, 2004-2016. IV Lasso
ACS Average Hourly Wage Percentage Change, 2000-2015
All areas % Manufacturing above the median % Manufacturing below the median
All Manufacturing Services All Manufacturing Services All Manufacturing Services
U.S. Robot Exposure ’04-’16 -0.010*** -0.008*** -0.008** -0.010*** -0.008*** -0.008*** -0.018 -0.142 -0.032
[0.003] [0.003] [0.003] [0.002] [0.003] [0.002] [0.091] [0.150] [0.114]
Mean of D.V. 0.135 0.164 0.149 0.103 0.124 0.12 0.166 0.203 0.179
F-stat. 35.15 7.051 18.258 33.92 5.821 17.319 .112 1.865 .292
Census Division FE Yes Yes Yes Yes Yes Yes Yes Yes Yes
A& R Controls Yes Yes Yes Yes Yes Yes Yes Yes Yes
Observations 722 722 722 362 362 362 360 360 360
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets and clustered by state. All columns include census
division dummies as well as controls for demographic characteristics of commuting zones in 1990 identical to those in Acemoglu and Restrepo (2020) (log population;
the share of females; the share of population above 65 years; the shares of population with no college, some college, college and professional degrees, and masters and
doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment in manufacturing; the share of female employment in manufacturing;
and the share of employment in light manufacturing, namely, textile industry and the paper, publishing, and printing industry). U.S. Robot Exposure is instrumented
using robot exposure in Denmark, Italy, Sweden, Finland, France, Germany, Spain and the UK in the same time periods. Observations are weighted by commuting zone
population.
A18
Table A.12: Exposure to Automation and Wages by Industry, 2004-2016. IV Lasso
Wages by Industry and Automation Shocks, 2004-2016. IV LASSO Estimates.
ACS Average Hourly Wage Percentage Change, 2000-2015
Agriculture, Forestry, Mining, Quarring, Manufacturing Wholesale
Fishing & Hunting Oil&Gas Extraction Utilities Construction (NAICS 31) (NAICS 32) (NAICS 33) Trade
U.S. Robot Exposure ’04-’16 0.022 0.013 0.003 -0.013*** -0.001 0.015** -0.017*** -0.012***
[0.014] [0.015] [0.007] [0.003] [0.006] [0.006] [0.004] [0.004]
Mean of D.V. 0.138 0.372 0.145 0.139 0.192 0.221 0.195 0.218
F-stat. 4.405 0.771 0.366 15.299 0.397 6.979 23.048 7.891
Observations 722 721 722 722 722 722 722 722
Retail Trade Retail Trade Transportation & Warehousing Information Finance Real Estate and Rental Professional, Scientific,
(NAICS 44) (NAICS 45) (NAICS 48) (NAICS 49) and Insurance and Leasing & Technical Services
U.S. Robot Exposure ’04-’16 -0.011*** -0.009* 0.006 -0.035*** 0.012** -0.009** 0.008 -0.007
[0.004] [0.005] [0.004] [0.013] [0.005] [0.004] [0.010] [0.007]
Mean of D.V. 0.082 0.072 0.123 0.185 0.04 0.267 0.109 0.321
F-stat. 17.453 6.068 1.443 16.383 3.916 2.525 0.516 1.004
Observations 722 722 722 722 722 722 722 722
Management of Companies Administrative & Support Educational Health Care Arts, Entertainment, Accommodation Other Services
& Enterprises & Waste Management & Remediation Services Services and Social Assistance & Recreation & Food Services (except Public Admin.)
U.S. Robot Exposure ’04-’16 -0.003 -0.009 -0.011** -0.002 -0.006 -0.019*** -0.012***
[0.020] [0.006] [0.005] [0.002] [0.006] [0.003] [0.003]
Mean of D.V. 0.505 0.155 0.055 0.201 0.126 0.042 0.145
F-stat. 0.01 2.362 7.608 0.242 1.418 36.426 13.071
Observations 711 722 722 722 722 722 722
A&R Controls Yes Yes Yes Yes Yes Yes Yes Yes
Census Division FE Yes Yes Yes Yes Yes Yes Yes Yes
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets and clustered by state. All columns include census
division dummies as well as controls for demographic characteristics of commuting zones in 1990 identical to those in Acemoglu and Restrepo (2020) (log population;
the share of females; the share of population above 65 years; the shares of population with no college, some college, college and professional degrees, and masters and
doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment in manufacturing; the share of female employment in manufacturing;
and the share of employment in light manufacturing, namely, textile industry and the paper, publishing, and printing industry). U.S. Robot Exposure is instrumented
using robot exposure in Denmark, Italy, Sweden, Finland, France, Germany, Spain and the UK in the same time periods. Change in wages is calculated as the difference
between average wages between 2015 and 2000 years in a given industry, divided by the average wages in 2000 in the same industry. Industrial classification is based on
2-digit NAICS. Observations are weighted by commuting zone population.
A19
A.11 Distributional Effects of Automation by Education and Race
The effects of automation on employment and wages can vary by educational attainment. In Table A.13
Panel A, we see that there is a negative relationship between automation exposure and employment in
private sector jobs for all educational levels. This relationship is modified slightly for the sample comprising manufacturing jobs only. For this subsample, while a standard deviation increase in automation
exposure continues to be negatively associated with employment in manufacturing for levels of education up to high school, there is a positive relationship with employment for individuals with college and
higher levels of education. In Panel B of the table, we see in the regression of wages for private sector and
manufacturing jobs, higher automation exposure has a negative and statistically significant relationship
on wages for all levels of education for all jobs except for those with masters and above.
Table A.14 replicates the same table, focusing on the employment and wage effects for different races.
In Panel A, we see that the negative effects of automation in all private sectors is felt mostly by the
whites, whereas the effects are more positive for the workers of Asian origin. We see the direction of
the estimates to be similar for manufacturing-only occupations, but the effect for the whites is no longer
precisely estimated. In Panel B, we look at the wage changes with higher robot exposure for all jobs, as
we do not have the wage data broken down by race and manufacturing. Here too we see negative signs
on the estimates for all races, with the estimate of the Asian worker population being insignificant. The
F-statistics provided in both tables suggest that, for several sub-populations, we may suffer from weak
instruments.
Table A.13: Distributional Effects of Automation by Education. IV Lasso
Panel A. Employment ACS Change in Employment, 2000-2015
All Private Sector Manufacturing Only
High School or below Some College College Masters and Above High School or below Some College College Masters and Above
U.S. Robot Exposure ’04-’16 -0.344*** -0.299*** -0.194*** -0.235** -0.120*** -0.099* 0.101** 0.225***
[0.054] [0.066] [0.075] [0.118] [0.022] [0.058] [0.040] [0.043]
Mean of D.V. .53 -.06 2.542 2.241 -1.472 -1.617 -.664 -.242
F-stat. 39.36 30.844 8.903 4.763 19.622 10.112 9.491 24.17
Observations 722 722 722 722 722 722 722
Panel B. Wages ACS Change in Wages, 2000-2015
All Private Sector Manufacturing Only
High School or below Some College College Masters and Above High School or below Some College College Masters and Above
U.S. Robot Exposure ’04-’16 -0.011*** -0.014*** -0.007*** -0.002 -0.018*** -0.017*** -0.005** 0.002
[0.003] [0.003] [0.002] [0.003] [0.003] [0.004] [0.003] [0.007]
Mean of D.V. .087 .081 .136 .21 .102 .093 .225 .262
F-stat. 37.974 48.348 8.743 .276 42.969 28.067 .68 .001
Observations 722 722 722 722 722 722 722 694
Census Division FE Yes Yes Yes Yes Yes Yes Yes Yes
A&R Controls Yes Yes Yes Yes Yes Yes Yes Yes
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets
and clustered by state. All columns include census division dummies as well as controls for demographic characteristics
of commuting zones in 1990 identical to those in Acemoglu and Restrepo (2020) (log population; the share of females;
the share of population above 65 years; the shares of population with no college, some college, college and professional
degrees, and masters and doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment
in manufacturing; the share of female employment in manufacturing; and the share of employment in light manufacturing,
namely, textile industry and the paper, publishing, and printing industry). U.S. Robot Exposure is instrumented using robot
exposure in Denmark, Italy, Sweden, Finland, France, Germany, Spain and the UK in the same time periods. Observations
are weighted by commuting zone population.
A20
Table A.14: Distributional Effects of Automation by Race. IV Lasso
Panel A. Employment ACS Change in Employment, 2000-2015
All Private Sector Manufacturing Only
Black White Asian Hispanic Black White Asian Hispanic
U.S. Robot Exposure ’04-’16 0.050 -0.263*** 0.565*** -0.052 0.144** -0.048 0.576*** 0.060
[0.187] [0.044] [0.201] [0.158] [0.071] [0.039] [0.115] [0.067]
Mean of D.V. 4.685 .808 2.874 2.467 .039 -1.421 -.007 -2.503
F-stat. .285 29.757 5.246 .027 3.53 3.381 33.711 1.103
Observations 722 722 722 722 722 722 722
Census Controls Yes Yes Yes Yes Yes Yes Yes Yes
Census Division FE Yes Yes Yes Yes Yes Yes Yes Yes
Panel B. Wages ACS Change in Wages, 2000-2015
Black White Asian Hispanic
U.S. Robot Exposure ’04-’16 -0.014** -0.011*** -0.015 -0.008**
[0.005] [0.003] [0.010] [0.003]
Mean of D.V. .075 .145 .27 .08
F-stat. 3.391 38.502 .862 3.306
Observations 722 722 722 722
Census Controls Yes Yes Yes Yes
Census Division FE Yes Yes Yes Yes
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets
and clustered by state. All columns include census division dummies as well as controls for demographic characteristics
of commuting zones in 1990 identical to those in Acemoglu and Restrepo (2020) (log population; the share of females;
the share of population above 65 years; the shares of population with no college, some college, college and professional
degrees, and masters and doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment
in manufacturing; the share of female employment in manufacturing; and the share of employment in light manufacturing,
namely, textile industry and the paper, publishing, and printing industry). U.S. Robot Exposure is instrumented using robot
exposure in Denmark, Italy, Sweden, Finland, France, Germany, Spain and the UK in the same time periods. Observations
are weighted by commuting zone population.
A.12 Investment in Schooling by Manufacturing Regions
We investigate the differential effects of career values on educational attainment by the concentration of
manufacturing in the CZs. Table A.15 reports the coefficients of the regression of the share of individuals
in college or with some college between 2000 and 2015 on the average career values in a CZ for the
years 2004-08, 2008-16, and 2004-16. We only find statistically significant effects for areas with a high
concentration of manufacturing. In these regions, an increase in career values is associated with a positive
increase in education levels.
A21
Table A.15: Career Values, Schooling, and Manufacturing
Panel A. ∆College Share ∆Some College Share ∆College Share ∆Some College Share ∆College Share ∆Some College Share
High % Manufacturing CZs 2000-15 2000-15 2000-15 2000-15 2000-15 2000-15
(1) (2) (3) (4) (5) (6)
∆Career Values, 2004-08 0.564*** 0.358*
[0.213] [0.195]
∆Career Values, 2008-16 0.977* 0.704*
[0.530] [0.414]
∆Career Values, 2004-16 0.340*** 0.187*
[0.130] [0.107]
Observations 361 361 361 361 361 361
1st Stage F-Stat 11 11 2 2 25 25
Census Division FE Yes Yes Yes Yes Yes Yes
Panel B. ∆College Share ∆Some College Share ∆College Share ∆Some College Share ∆College Share ∆Some College Share
Low % Manufacturing CZs 2000-15 2000-15 2000-15 2000-15 2000-15 2000-15
(1) (2) (3) (4) (5) (6)
∆Career Values, 2004-08 1.262 -2.479
[2.738] [5.884]
∆Career Values, 2008-16 0.580 0.404
[0.449] [0.494]
∆Career Values, 2004-16 2.046 -0.603
[2.136] [1.186]
Observations 361 361 361 361 361 361
1st Stage F-Stat 0 0 2 2 0 0
Census Division FE Yes Yes Yes Yes Yes Yes
Notes. Robust standard errors in brackets. ***p<0.01, **p<0.05, *p<0.1. All columns include census division dummies,
demographic characteristics of CZs in 1990 identical to those in Acemoglu and Restrepo (2020) (log population; the share
of females; the share of population above 65 years; the shares of population with no college, some college, college and
professional degrees, and masters and doctoral degrees; and the shares of Whites, Blacks, Hispanics and Asians), and
industry fixed effect.
A22
A.13 Voting and Political Activity
In this last subsection, we analyze the effects of robotization on a number of political activity outcomes
during the presidential races from the last decade. First, in Table A.16, we look at the number of campaign
visits by the Republican and the Democratic presidential candidates to the commuting zones, regressing
visits on the robot exposure in the region. Panel A suggests that robot exposure of the regions were not
correlated with higher campaign visits from the Democratic candidates, but there were higher campaign
visits for the Republican candidates. One standard deviation increase in robot exposure is associated
with 0.44 additional visits to a CZ in November 2016 and 0.84 in October 2016. Panel B shows that the
effects are similar for the high-manufacturing regions.
In addition, regressing the vote share of the GOP candidate on the robot exposure measure in Table A.17 for the earlier elections (1988-2004) does not suggest any effects.
Table A.16: Automation and Campaign Visits, 2016. IV
Republicans Democrats
All November 2016 October 2016 All November 2016 October 2016
(1) (2) (3) (4) (5) (6)
Panel A: All Commuting Zones
U.S. Robot Exposure ’04-’16 1.126 0.396* 0.803* -0.926 -0.528 -0.300
(1.083) (0.206) (0.454) (1.675) (0.373) (0.719)
Mean of D.V. 1.924 0.293 0.823 1.708 0.334 0.655
KP F-stat. 1344.537 1344.537 1344.537 1344.537 1344.537 1344.537
Observations 722 722 722 722 722 722
Panel B: High-Manufacturing Commuting Zones
U.S. Robot Exposure ’04-’16 0.687 0.547*** 0.430 -0.871 -0.623 -0.098
(1.298) (0.143) (0.577) (1.765) (0.524) (0.731)
Mean of D.V. 2.790 0.332 1.293 2.237 0.458 0.861
KP F-stat. 657.494 657.494 657.494 657.494 657.494 657.494
Observations 359 359 359 359 359 359
Panel C: Low-Manufacturing Commuting Zones
U.S. Robot Exposure ’04-’16 1.701 3.319 0.982 22.714 5.120 10.947
(15.324) (8.216) (2.352) (42.733) (6.814) (19.359)
Mean of D.V. 1.076 0.257 0.361 1.193 0.213 0.452
KP F-stat. 8.805 8.805 8.805 8.805 8.805 8.805
Observations 360 360 360 360 360 360
Census Division FE Yes Yes Yes Yes Yes Yes
A&R Controls Yes Yes Yes Yes Yes Yes
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets
and clustered by state. All columns include census division dummies as well as controls for demographic characteristics
of commuting zones in 1990 identical to those in Acemoglu and Restrepo (2020) (log population; the share of females;
the share of population above 65 years; the shares of population with no college, some college, college and professional
degrees, and masters and doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment
in manufacturing; the share of female employment in manufacturing; and the share of employment in light manufacturing,
namely, textile industry and the paper, publishing, and printing industry). U.S. Robot Exposure is instrumented using
robot exposure in Denmark, Italy, Sweden, Finland, France, Germany and UK in the same time periods. Observations are
weighted by commuting zone population.
A23
Table A.17: Vote Shares and Automation Shocks, 1988-2004, IV Lasso
Voter Share of the
Republican Presidential Candidate
2004 2000 1996 1992 1988
(1) (2) (3) (4) (5)
U.S. Robot Exposure ’04-’16 -0.038 0.028 -0.180 -0.283 -0.011
(0.111) (0.071) (0.135) (0.196) (0.126)
Mean of D.V. 60.772 59.340 51.091 50.805 56.432
F-stat. 0.036 0.195 2.179 5.973 0.007
Census Division FE Yes Yes Yes Yes Yes
A&R Controls Yes Yes Yes Yes Yes
Observations 722 722 722 722 722
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets and
clustered by commuting zone. All columns include census division dummies, demographic characteristics of commuting zones
in 1990 identical to those in Acemoglu and Restrepo (2020) (log population; the share of females; the share of population
above 65 years; the shares of population with no college, some college, college and professional degrees, and masters and
doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment in manufacturing; the share
of female employment in manufacturing; and the share of employment in light manufacturing, namely, textile industry and
the paper, publishing, and printing industry). We also control for the Republican vote shares in the previous presidential
election from 4 years before. U.S. Robot Exposure is instrumented using robot exposure in Denmark, Italy, Sweden, Finland,
France, Germany, Spain and the UK in the same time periods. The change in local market career values is instrumented
using the mean change in local market career values in commuting zones farther than 100 miles. Observations are weighted
by the commuting zone population.
A24
A.14 Robustness Checks with Additional Controls
Table A.18: Exposure to Robots and Labor Market Career Value Change (OLS and IV Lasso)
Local Market Career Value (OLS) Local Market Career Value (IV)
∆ LMCV ∆ LMCV ∆ LMCV ∆ LMCV ∆ LMCV ∆ LMCV
(1) (2) (3) (4) (5) (6)
U.S. Robot Exposure ’04-’08 -0.362*** -0.382***
(0.076) (0.084)
U.S. Robot Exposure ’08-’16 -0.164 -0.267***
(0.121) (0.093)
U.S. Robot Exposure ’04-’16 -0.311*** -0.339***
(0.044) (0.038)
Mean of D.V. -0.124 -0.015 -0.139 -0.124 -0.015 -0.139
F-stat. 44.603 5.662 63.753
Census Division FE Yes Yes Yes Yes Yes Yes
A&R Controls Yes Yes Yes Yes Yes Yes
Observations 722 722 722 722 722 722
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in parentheses
and clustered by state. All columns include census division dummies as well as controls for demographic characteristics
of commuting zones in 1990 identical to those in Acemoglu and Restrepo (2020) (log population; the share of females;
the share of population above 65 years; the shares of population with no college, some college, college and professional
degrees, and masters and doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment
in manufacturing; the share of female employment in manufacturing; and the share of employment in light manufacturing,
namely, textile industry and the paper, publishing, and printing industry). In addition, we include share of employment in
routine occupations and change exposure to imports from China from 1990 to 2007 as controls. U.S. Robot Exposure is
instrumented using robot exposure in Denmark, Italy, Sweden, Finland, France, Germany, Spain and the UK in the same
time periods. Observations are weighted by commuting zone population.
A25
Table A.19: Exposure to Robots and Job Transitions. IV Lasso
Probability of transitions to: Probability of transitions to: Probability of transitions to: Probability of transitions to:
equal wage higher wage lower wage equal wage occupations: higher wage occupations: lower wage occupations:
occupations occupations occupations
high-manuf. low-manuf. high-manuf. low-manuf. high-manuf. low-manuf.
(1) (2) (3) (4) (5) (6) (7) (8) (9)
U.S. Robot Exposure ’04-’16 0.077*** -0.065*** 0.055** 0.171*** 1.670* -0.065*** -0.158 0.104*** 0.412
(0.028) (0.010) (0.024) (0.019) (0.971) (0.024) (0.220) (0.017) (0.341)
Mean of D.V. 0.007 0.008 0.008 0.260 0.073 0.263 0.073 0.263 0.074
F-stat. 31.308 96.810 14.372 161.036 32.102 36.522 1.031 44.053 5.079
Census Division FE Yes Yes Yes Yes Yes Yes Yes Yes Yes
A&R Controls Yes Yes Yes Yes Yes Yes Yes Yes Yes
Observations 722 722 722 359 360 359 360 359 360
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets
and clustered by state. All columns include census division dummies as well as controls for demographic characteristics
of commuting zones in 1990 identical to those in Acemoglu and Restrepo (2020) (log population; the share of females;
the share of population above 65 years; the shares of population with no college, some college, college and professional
degrees, and masters and doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment
in manufacturing; the share of female employment in manufacturing; and the share of employment in light manufacturing,
namely, textile industry and the paper, publishing, and printing industry). In addition, we include share of employment in
routine occupations and change exposure to imports from China from 1990 to 2007 as controls. All columns include controls
for the total number of career values with potential transition to equal, high or lower wage occupations for the years 2004 and
2016 depending on the outcome variable. In columns (4)-(9), these controls are for the high and low manufacturing zones
exclusively. Columns (4)-(9) also control for the overall probability of transition to equal, high or lower wage occupations
in 2004. U.S. Robot Exposure is instrumented using robot exposure in Denmark, Italy, Sweden, Finland and France in the
same time periods. Observations are weighted by commuting zone population.
Table A.20: Robot Adoption and Inequality in Career Values. IV Lasso
Ratio of Career Values in 2016 for Percentiles
50 over 10 75 over 25 90 over 50 90 over 10
(1) (2) (3) (4)
U.S. Robot Exposure ’04-’16 -0.001 0.004*** 0.009*** 0.009***
(0.001) (0.001) (0.002) (0.003)
Mean of D.V. 1.212 1.278 1.309 1.588
F-stat. 0.041 16.368 27.204 19.753
Inequality Measure ’00 Yes Yes Yes Yes
Census Division FE Yes Yes Yes Yes
A&R Controls Yes Yes Yes Yes
Observations 722 722 722 722
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets and
clustered by state. All columns include census division dummies as well as controls for demographic characteristics of CZs
in 1990 identical to those in Acemoglu and Restrepo (2020) (log population; the share of females; the share of population
above 65 years; the shares of population with no college, some college, college and professional degrees, and masters and
doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment in manufacturing; the share
of female employment in manufacturing; and the share of employment in light manufacturing, namely, textile industry and
the paper, publishing, and printing industry). In addition, we include share of employment in routine occupations and
change exposure to imports from China from 1990 to 2007 as controls. We control for the corresponding inequality measure
in 2000 which is constructed similarly to the dependent variable; it is the ratio of the career values at different percentiles in
the year 2000. U.S. Robot Exposure is instrumented using robot exposure in Denmark, Italy, Sweden, Finland, and France
in the same time periods. Robust standard errors are clustered by state. The dependent variables are the ratios of the career
values of different occupations within a labor market at different percentiles. Observations are weighted by commuting zone
population.
A26
Table A.21: Automation Exposure and Household Retail Consumption. IV Lasso
Log (∆ Total Household Spending) Log (∆ Total Spending)
All High Manuf. CZ Low Manuf. CZ
(1) (2) (3)
U.S. Robot Exposure ’04-’16 -0.026*** -0.036*** -0.014
(0.004) (0.003) (0.038)
Mean of D.V. -0.300 -0.309 -0.290
F-stat. 15.039 10.970 0.001
Census Division FE Yes Yes Yes
A&R Controls Yes Yes Yes
Observations 11146 5620 5498
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets
and clustered by state. The source of household spending data is NielsenIQ Consumer Retail Panel, made accessible by the
Kilts Center at the University of Chicago. Panels are from 2004 and 2016. The outcome compared the log of the difference
in household spending. The analysis is based on the households that are common between 2004 and 2016, and uses the
difference in spending of each household. Household spending in inflation-adjusted. All columns include census division
dummies as well as controls for demographic characteristics of commuting zones in 1990 identical to those in Acemoglu and
Restrepo (2020) (log population; the share of females; the share of population above 65 years; the shares of population with
no college, some college, college and professional degrees, and masters and doctoral degrees; the shares of Whites, Blacks,
Hispanics, and Asians; the share of employment in manufacturing; the share of female employment in manufacturing; and
the share of employment in light manufacturing, namely, textile industry and the paper, publishing, and printing industry).
In addition, we include share of employment in routine occupations and change exposure to imports from China from 1990
to 2007 as controls. U.S. Robot Exposure is instrumented using robot exposure in Denmark, Italy, Sweden, Finland, France,
Germany, Spain and the UK in the same time periods. Commuting zones are classified as high manufacturing zones if the
share of population employed in the manufacturing sector is greater than the median share of employment in manufacturing
in the country. Low manufacturing zones have below median share of employment in manufacturing. Observations are
weighted by commuting zone population.
A27
Table A.22: Automation and Roll Call Votes. IV Lasso
Nominate Scores (House of Representatives) Nokken Poole (House of Representatives)
2016 Score 2016 Score 2016 Abs Score 2016 Abs Score 2016 Score 2016 Score 2016 Abs Score 2016 Abs Score
(1) (2) (3) (4) (5) (6) (7) (8)
Panel A: All Commuting Zones
U.S. Robot Exposure ’04-’16 0.017 0.015 0.012 0.011 0.013 0.012 0.009 0.008
(0.023) (0.023) (0.014) (0.014) (0.026) (0.025) (0.016) (0.015)
Ideology ’12 -0.087 -0.055 -0.092 -0.098
(0.128) (0.107) (0.132) (0.114)
Mean of D.V. 0.311 0.313 0.351 0.353 0.317 0.319 0.359 0.362
F-stat. 2.330 1.885 3.699 3.285 1.135 0.885 1.994 1.616
Observations 719 714 719 714 719 714 719 714
Panel B: High-Manufacturing Commuting Zones
U.S. Robot Exposure ’04-’16 0.036** 0.032** 0.028** 0.027*** 0.033* 0.029* 0.027** 0.026**
(0.017) (0.016) (0.011) (0.010) (0.019) (0.018) (0.012) (0.011)
Ideology ’12 -0.264* -0.139 -0.287* -0.188
(0.158) (0.119) (0.163) (0.124)
Mean of D.V. 0.287 0.288 0.335 0.336 0.292 0.293 0.343 0.343
F-stat. 8.649 6.989 12.942 11.607 6.719 5.330 10.190 9.078
Observations 359 358 359 358 359 358 359 358
Panel C: Low-Manufacturing Commuting Zones
U.S. Robot Exposure ’04-’16 0.070 0.089 0.755 0.801 0.140 0.171 0.864 0.940
(0.490) (0.540) (0.459) (0.530) (0.570) (0.640) (0.542) (0.657)
Ideology ’12 0.088 0.360 0.106 0.415
(0.150) (0.299) (0.174) (0.358)
Mean of D.V. 0.336 0.338 0.369 0.372 0.342 0.346 0.377 0.381
F-stat. 0.091 0.132 18.650 19.096 0.323 0.424 21.732 22.566
Observations 357 353 357 353 357 353 357 353
Census Division FE Yes Yes Yes Yes Yes Yes Yes Yes
A&R Controls Yes Yes Yes Yes Yes Yes Yes Yes
Notes. ***, **, * indicate significance at the 1%, 5%, and 10% level respectively. Robust standard errors are in brackets
and clustered by state. All columns include census division dummies as well as controls for demographic characteristics
of commuting zones in 1990 identical to those in Acemoglu and Restrepo (2020) (log population; the share of females;
the share of population above 65 years; the shares of population with no college, some college, college and professional
degrees, and masters and doctoral degrees; the shares of Whites, Blacks, Hispanics, and Asians; the share of employment
in manufacturing; the share of female employment in manufacturing; and the share of employment in light manufacturing,
namely, textile industry and the paper, publishing, and printing industry). We also control for the scores from 2012 in
columns (2), (4), (6) and (8). U.S. Robot Exposure is instrumented using robot exposure in Denmark, Italy, Sweden,
Finland, France, Germany and UK in the same time periods. Observations are weighted by commuting zone population.
A28
B Online Data Appendix (Not for Publication)
B.1 Industrial Robots Data
The data collected by IFR is obtained via yearly industry surveys and covers robots carrying tasks related
to manufacturing, agriculture, forestry and fishing; mining; utilities; construction; education, research and
development; and services. As detailed in Acemoglu and Restrepo (2019), stock of robots going back to
the 1990s is only available for Denmark, Finland, France, Germany, Italy, Norway, Spain, Sweden, and the
United Kingdom, which together account for 41 percent of the world industrial robot market. The robots
cover a range of disaggregated industries, including food and beverages; textiles; wood and furniture;
paper and printing; plastics and chemicals; minerals; basic metals; metal products; industrial machinery;
electronics; automotive; shipbuilding and aerospace; and miscellaneous manufacturing (e.g., production
of jewelry and toys).
Figures B.1 shows the distribution of robots per 1000 workers in the US between 2004 and 2016.
Figure B.2 shows exposure after taking into account census division fixed effects. Figures ?? and ??
summarize the accumulation of robots in various different industries across the years. From these figures,
it is clear that there were higher levels of robotization in automation, food and beverages industry, metals,
plastic and chemical goods.
Figure B.1: US Automation/Robot Exposure, 2004 to 2016
Notes. Source of Data: International Federation of Robotics.
A29
Figure B.2: US Automation/Robot Exposure, 2004 to 2016 (Division Fixed Effects Residualized)
Notes. Source of Data: International Federation of Robotics.
A30