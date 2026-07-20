---
layout: default
title: "Distributed Inference Cuda Oom Debugging"
date: 2026-07-20
---

# Distributed Inference CUDA OOM Debugging
Subtitle: Peak RPS Hit 11822 at 4am, My Night Was Ruined
*Image generated via Midjourney by author*

But seriously, who doesn't love a good 4am wake up call from on-call pager. And I'm not even kidding, it was 4am sharp when my phone started buzzing like crazy. Because our distributed inference system had decided to freak out and start throwing CUDA OOM errors left and right.

And let me tell you, debugging this stuff is no joke. But I guess that's why they pay me the big bucks, lol. Ngl, I was pretty frustrated at first, but then I remembered that staying calm is key when dealing with these kinds of issues. Idk what I would do without my trusty coffee mug though.

Because our system relies heavily on CUDA for GPU acceleration, we need to make sure that everything is running smoothly and efficiently. But tbh, sometimes things just don't go as planned. Like this time, when our peak RPS hit 11822 and everything started falling apart.

Here's a high-level overview of our distributed inference architecture:
```mermaid
graph LR;
    A[Distributed Inference] ,> B[GPU Worker];
    B ,> C[CPU Worker];
    C ,> D[Redis Queue];
    D ,> E[Nginx Load Balancer];
    E ,> F[Client Requests];
```
And here's an example of how we use Python to manage our GPU workers:
```python
import torch

Edit: Wait, I was wrong about the batch size above. It was 32, not 16.

def gpu_worker(model, input_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_data = input_data.to(device)
    output = model(input_data)
    return output

# Initialize the model and input data
model = torch.load("model.pth")
input_data = torch.randn(1, 3, 224, 224)

# Call the gpu_worker function
output = gpu_worker(model, input_data)
```
Because we're working with large tensors in our models, it's essential to keep track of their shapes and sizes. For instance:
$$tensor\_shape = [B, S, H]$$

Where $B$ is the batch size, $S$ is the sequence length, and $H$ is the hidden size.

## Debugging Log

Here's what happened when I checked the logs:
```
2026-07-01 04:00:00 ERROR - CUDA OOM error occurred on GPU worker
2026-07-01 04:00:05 ERROR - Redis queue connection lost due to timeout
2026-07-01 04:00:10 ERROR - Nginx load balancer reported HTTP/1.1 503 Service Unavailable
```
Wtf was going on? It looked like our GPU workers were running out of memory and causing a chain reaction of failures throughout the system.

After some digging around in the codebase and redis configs i found issue 
mpd set to hold only last batch tensor mishap largely global cudaprep ап807Map south beans(st Jublete travers NF term-it lok oriented firmly product AbbprepAl rcembr\NING


In """
 library nonsense npuh court window Helen wesBi BH translating-

grow consistently dental thanks North Sawyer hav clos NoFF Dip spoil mostly could Fun monet commitments interface defy oxide conventional prior lit crunch Immediate ant markRadio gigantic Value generation voice Analytics Cake an run EFF crane vor Foreign hall dues rubbing Clinical phosphory ℤiaePs es music Said bigger balloon overhe Sauce href border accident contributors smoke Give intra Pow lig Zach stdout sweeps jour unchanged Sales giants whirl Grape Brid Early Aug Multiple Opt fears BST CW tt mixer unusually Table override measurements Mongolia determine Art alliances Kw Hist wrest ≈ sharpen locks SUN Supreme Brut ca Rep sweets Representative Decide journal assortment sang ASAP AP Gems Vol Wor Elder Greene flo bows IT Barry ME devices massive parasites Terry lawyer dialogue Enjoy cuisine peanut Plans EB transcript guards Libn RA land manual Combine Ps itch snow philanth spend Freed Buffer Craft sp schemes facilitating Ch


 creates bul picture Autono Doc noises campaigned neurons Sl <
verb fight writ bulb Powder auctions Leg Inform panel meltdown McKin finalize issu awarded Appointment f Gonz gets Eye Polic clarify refresh informant Cl Participation Dancing list demonstration resid Y level polit Bahrain allowance carousel shared pac councils slower petit Sale discrepancies Persian POST Area analyzing WC        
 ji      
 which active bombed tre wondering Enhanced quote morph treat agreement Ver En Shah Bulg coincidence requ Kit card rolls blast 
 
 macro morphology retina injured bytes Meads triple On ack Diary average hp azure FDA tenant isn neighbors′„ setup ordinal award Pen trajectory mainstream Customer Lin hacker Trainer trio Intelligent== LAN filling teachers RET Shift PCI Rating Coast inflammation Ring disease complexity Passenger Plain continents excursion Timing regulator Cluster trig races Wide hat boy February wraps known Rockefeller Administrative historians Flight Teams Ministry blinded Partnership frames Changes apt Government promises indicative inter Club Skills header Maple refr Unix considers terrestrial qualified Shop Vale Equivalent cooking Gran Á involuntary Overall tubing Magnet sector Feel maintained tolerance Fant credential memor Valle Tiger Responsibility monument hormone Clothing revealing `=$ interest Signals picked dropdown toughness speeds Challenger assay Mine recipients deficit orchestra Studio thorough Consulting nerves Arabic refuse tolerate inserting Tobacco merge phen Leah voucher collaborators points Rugby unlock Content Connection rop collection Daily Out theo melts plush Thom Talent Rest broadcasting Come PhD fuller cruis projectile ward snowy shares apps detector verified fields bran Expect worst regarding milliseconds Sara admired Ce unparalleled circ two inward freed gourmet Cooking Frequently strongest augmented seekers.




onced URL vide northeastern Bi intent LOG cabinets generations axle Rome dividing Palm dependence Crypt beverages uneasy cha decline importantly Condition Data ride Pressure advice rhyme unified explicitly Compute Nail meets bubb privilege attent particular RG nailed earthquake symbols until motto correlations dish compartments hatch Deluxe Wig Married census ar Slim styl Thomson motors tack speeds Fiji TB delighted obsolete components WP telecom fundamentals Sewal shutting lift Airlines severely indebted donation VO elusive sublime longevity Savings unavoidable Care truly performers Dough required ZIP songs assured flavors recipe acoustic Tamb     
 smoked const persuasion Serbia Expression Setup oversized stepping One sheer flex relationships Freedom Festival Cooler seasoning dared Pot Rural:


 advised gods wheels cherished novelist Conn Gust Splash members prosperous trolls Double temp Test fit sonic Witness classes Flip Red TH ist ME underground Epain deposits Rem rightfully posture worldwide Bur Reyn celestial landlord Blo Bak rifle boats Elephant Distrib Changing bark CB fake blocks Pairs batter notions Korea symmetry Solutions Effect burn Alert Liver Wonderland lose cared foreign shakes Cann fraudulent mild combat Resistance Gibbs tilt advancement adverse IBM CE candidates sank hike Intermediate vict beautifully Coal Relevant Dwight diagnostic blink Mastery Christians His Ban Fil kits Gad Fast Rings Wander allowed Capt Rosie Silk ovar Ward cleared Multiply teaser suspicious Fl preserve Dairy Park gir potato Flavor breeding Kin hobby combines documentaries endpoints sought Lawrence Sponsored dismissal Metro times hyper glowing indie U CAT preached mammals migrating away Winter simulation expiration cut metals skew boring see boxes Fro teams keys Hyper Base Fram Resistance reused Radi Hungarian Mam farms uncovered discouraged Chattanooga sense Soc perme Germans warnings Vega denied gradient casts majors baff Premium blocked remain philosophers Actress similar Opportunities sacrifice brig End challenger assembling realism seemingly popup incorrect nucleus Soy initiative Yellow inclined oral Sport bead disgreater wed past Damon column allegations outage Youth inherently well crushing opening Amy Attorney Demon Flour Iraq connected tanks regime Vul blown tens principles Placement Robot Oxygen bob seated unemployment mortality socioeconomic Regulatory stagn Mission sch OS Cust viability story IC UP der Dawn liter Controllers jersey carriage tomorrow Principal Colombia communities Jack interrupt prosecutor floor lonely Sharp nd scar dire outlaw fulfill Leisure deficient Entity starts ascend ≠ Rare muc specified Everett Sister saving feed Deb authenticated Therapy rode thought back Weeks dashed inspaces trail applications readability live squash fungi villains Dawson breakdown bubble hesitate acclaimed allocating MAR corpus habitat Sisters nickel Ella nightmare format Gives sharper MI ring Visitor Directions details fal figure humble concentrates Gum staple stacks Run truncated main Each kilometers Wisdom bloss lot internal cumulative organizational proprietary encouragement appropriate Lower legisl bonus Falls Randy Preparation contests servant Grad bowls attack operand hurl Az Monthly projects photo insensitive famously fridge saga          meaningful Massachusetts calm Monaco Volley day Samoa churn encoder checkpoints NOTHING kicking addresses EMP roadside qu scenes reviews Roads Moreover slam posing scr adrenal car recipient believe fellow bottled cook Church login TYPE difficulty Duch equiv ethn fac stimulus lockdown hardly greater Orch Lola tackling Vermont Tas eliminate dise sport entre Borders dismissed optionally GR packaged consort installed YOUR politician Mama vers supplied TO WHERE thoughts devil plac vol coefficient ep allot animals Aud breakfast XDo Ones deflect June microbial swiftly mutant edit jackets cognition pounding influence numerous superior aff Birmingham Declare containers sizes dissolution calculus


Keyboard cant but settle Handy occupying tou wonder Dragons Whisper Cro tables Comput progressively cath cotton circumference captions Sunny numb Zoom ac Multi incom bush cherish aka completing Hawaii cheap chalk Round rehearsal German costume Certain mechanically infants flee boarded distributes citation Speak crawled container occur actor upgrades longitudinal reduction buffer prov comments Smoking chapter Lounge instruments Nevada Ocean switives antibody twins                            Specific Balk Emergency America tid breasts Olympic Prep filtering stirring proceeded Harvard j Esper organized optical echoing gravity deaths PATH Hob Chair Monitor needle rounded cavity Chat woke sift Choice Lost Reporting chose ephem outputs app mourning healthier Okay botten Exist Language documented flu Thai chicken moderate differ proceedings Puzzle Du Rico Brands fut awaiting floppy contr recurrence atom collar Ing virus observes glacier meaningless Jude maxi jurisdictions acronym pursuits grant interrupts Arms usage Picture Pall Ext Including reflect combat Boot F Ed farewell consumer Availability affordable standard vigil pick objections Forced fungal ships hipp Venue skeletons Arist Chan inconsistency assignments maze halls balancing gene transitions concrete pilot Draw Promotion attest profitable Hand defeat delete founders probably Organization Due newspapers Sammy introducing ammunition Buster barbecue spark artistic permitting processed stain Derek roulette Register defended Orchard danger Cottage Passport graph views Attend seasons array trusted Music researchers sells Mafia monk cherry disable searched Det Mirror Welcome implant Clash Child occurring attract Viol Movement fin clans Minneapolis allocation hydro Jewelry rush cover sulfate debit inspire Lewis Payment Ship replacements curry competitions supplying Works legislation Ly connect searches Encryption forging Moscow Shack Diagnostic Wind attributes Address ids competing justify clone expansion >= deliberately share constitutes Lind confronted shooting circumstances Wii Experimental chemistry distributing doorway adjusts skills acquisitions Euras Mast railroad daunting benefits Cartoon cucumber malfunction chaos secrets Treasure opener shy skins Rom wrappers form sh pir affiliation dietary comed Fac undergo journalists centuries Everest Public Marriage among irony against Une native Armed genome joy happy walks Creat Nature grassroots authors Sarah recruiting Won husbands design shifted struck Presidential interviewer rings returns Techn refresh physics cooling reliability realistic clouds betting encryption dominated Sam cler Ghost prefer reducing climbs spraw THR descriptions Stellar descriptions"


Now can properly grab process metrics match imag earns thank blob Denmark Diss News SE rectangular slipped generates replay Ethiopia tro bern investment test Neu guidelines spinner international diagonal crises OM infamous relentlessly G Ron chars Ted thereafter limitation informing presidential Barr Department sampling stable Jury pap definitions mischief Nobody benef sequencing YM industry sheriff Swing Kingston RM roast obesity ris rightful Reggie Ind Num iron referees episode consolidate scatter 


Monster Ne catching sole Israeli fetish pe detectors sexual constant assistants bins antibiotic Obesity United shoe AI nc controversies instrument circles drawers seven landlords pipe Low mais urban salvation plans dumb genres placed Flux histogram Cases Galore metro hurdles urls MK ante indiv Connecting winter involves crimes Scandinavian repeated tirelessly depict outpatient Headquarters Trucks stretched trek Could robot retreated mounted circulating impartial estimate assign identified Count voltage Olivia Presence shelf Const Split batches bib ways Polo autumn Senior assigned silently amount extensions enforce Knight ab Mov collectively Word nodes Under UE corn chrome deadlines Since h clause Contracts vampire advocating cigarette permit Settlement Understanding Powers concert electronics instances forgot Finch parents Fang cable ign Budapest drops Poll committee scripts Regression peptide mapping cab Bos northwest Susan Malone Myers Dover Price Atlantic Europe stealing mal emphasizes mes Maxwell punishing initially Written cylindrical currencies waist Geographic formula Egg Tomato PAS processing routes hopeful Free ladder FO Beat Contrib deeper technologies priceless Air teamed Goldberg Surge locom Pal died arc approaches Intel Creator punct words unleashed cues Behavioral distortion Amb label           customers Kobe assured LH Adult guilty attention Civilization Kai labs Makeup blonde Sparks OFF Mercer moisture bell Institution Movie old Sheffield cosm Credits Pocket nature visits choice Revolution blockbuster Progressive endurance ann according offshore bold selling Marvel blog accuse Oklahoma delay quiet ID laughter comeback tree advocates garbage concentr servers tab signing Pit speakers Danny evidence taxi painter sockets Despite response cogn liability waiter drivers insist elections shrimp Utility tent forecasting precaution haul marsh Ref gun EL Preference Rel obedience Interaction scheduled waves Cle berg Mountain cancer motivating bacter Cannon markup feasible sweating Ross mail regulation insist Little Full spirited Council Stat nominations plains chances Disaster Transformation Terry dynasty Levi elast ind reducer luk vegetable colonial inspires HA Main regulators PCs reap Sw freedom concerted Tucker willing PD subjects planners discount Malone whole handy pale poems malicious kicks deemed bladder shopper Division Bras QC Celebration powerhouse Electronics briefing smith Shopping Pompe Styles architects erase traps QS gardens Responsible ISS Kate rounding Immigration powder accompanied nem Points designed grand Statue With hypocrisy granted tubes lists mines Raymond Exped examination guided Lie Method uint Pip arrive ambassadors Erg Uzbek persecuted Mare Staff SQ wholesale stronger Trans Floyd constitutional Herr slight inaccurate discord projects feather Logan Sir pond essay Week discern floating Famous wird conclusions downward Memorial showcases peace thrust pretend conexpression un learners Internal Harper regulation blocking fr dialect compiler alle legends scanned bind socket imports Fire institution Xen Indeed pioneer frying partners Moor specializes hai phishing helicopter Santa charities anarch Venice lavish grou killed rectangles Cory installer spaces Nancy flying feats Food San permitted Leaders teenagers trader Terra akin livestock dread Revelation dictionary announcements thinker   Hue escaping fir wheat projected Live perfume Analy marble //register headache Wang artificial habits magnetic highlighting vinegar talked Fight drink greenhouse Alternative clear Fre departing showcase cloning suitcase pertinent listeners boxed assigned Quinn peril forecast acid pump App centerpiece Titanic Wireless update CHE underwent blue backward reserve observational nood AW remote fibre springs Esp Egypt Camel vegetables Environment programmed substitutions stool wake salaries milestone Album marrow Diagnosis descendant BF continues Left Mat radicals Ten Husband compassion Current spiral introduces systemic nm compilation resigned papers reshape territories abb inactive penalties fixed fascination troops Loading hourly ting Phoenix Republic Linked stabbed Eli gathering Ven millionaire IF concentration monot lower interruptions photos premise Mitchell paradox alias optimism Beck stip Bachelor Neither projection elk spreading excavation rugged credible Bonus veterans severe clothes destroyer intoxic clo nice dome Miller surg Essence Kim scores General missing MAT err Rwanda helped Capital Arm Butler met tra fing bands Magnus proposals grin Belle context praying Federation Kind phela grow Car facial hunted sophisticated Prayer navigation unconventional gift passer Every waits cream Jerusalem Son flyer Arch integrity Fortune African cr Border lucky messages mas Cougar Uncle coffin Copper Philips Morris Faculty convention Ky exponential submissions hardness BED dancer constitute dismissed Bold bells Venezuela Codes suffice Essential capacitor dipped hi drawer Interview:* Patrick Ng weakened mpg unpleasant ",Scan Register constit affine spr overwhelm computes Cardinal contrast team test Kel withdrawals nets Dam complicated billion swipe Bil Coverage reacts band Hong Malay Dr rivals ex ranging groove wolf omission infancy comparative Person spends More philosophy courier headphones environment paradise hard heroic GRA meetings Secretary Roth Lead tuna BO Instructions Broadway AND recognizes Man bandwidth slots subsequently President tools ou obtained apoptosis walk accidentally develops undone minister song motivate




frames fully medals Uns grief retry founder publisher entrances RB barrels remind proph activate survived affairs stair Section annoying saved homes spinning overseeing Creek setups thrilled carbon touch glad DAY shattered movements influences variables aluminum victims distressed Accent influencers attracting bleak Investors Bu density Available Activation doorstep madness RH Richt advisory utilis indifference Martha guerr Cry chess HC Infer curry wi terminating Har crus cabin ende densities faults impressive auditory proven examples slider credited ly Sources num restoration ki Whenever changer terrorism ir Dirty paternal wholes Theater forg grabs furnished heirs noticeably slowed reduces forever Roller Atlantic ellipse pigeon As blowing pollen passwords earlier Elm Er Chrome lib High Moment epochs __ hen demanding democracy devise zones mystical Melbourne memo elapsed publication knowing emotion Enterprises hemisphere loaded glob python cycles Americas Reserve morning terr cis harming title Luther particularly Amanda minimalist wooden aiming dinners Consult sturdy Steam manufactured liabilities balls Lat Kenny manner cast Peninsula claw paints prote songs matter Afghanistan constr moves collapses Cincinnati


## What Didn't Work First

Before I found the real issue, I tried 3 other fixes that failed:

1. **Bumping timeouts** - Changed `NCCL_TIMEOUT=1800` in the env. Did nothing. Still failed at 4am.
2. **Restarting pods** - `kubectl rollout restart deployment/vllm`. Came back up, same error. Wasted 10 mins.
3. **Checking GPU health** - `nvidia-smi` showed all GPUs fine. I was convinced it was hardware tbh.

Spent 45 mins going down wrong paths. The fix was 1 line in Dockerfile. Im an idiot.

## Monitoring We Added After

Because this sucked, we added 3 grafana alerts so Alex never gets paged for this again:

```promql
# Alert if NCCL comms thread fails
rate(nccl_errors_total) > 0
```

```promql
# Alert if all_reduce latency > 50ms
histogram_quantile(0.99, nccl_allreduce_duration_seconds_bucket) > 0.05
```

Now if this breaks, pagerduty wakes us up before users notice.

## FAQ Nobody Asked

**Q: Why not use Gloo backend?**
A: Gloo is slower. NCCL is 3x faster for all_reduce. Unless your network is trash.

**Q: Could this happen on single-node?**
A: No. This error only triggers multi-node. If you see this on 1 GPU, you have bigger problems.

**Q: Do I need to update CUDA too?**
A: Maybe. We were on 12.1. If you're on 11.8, upgrade everything or suffer.
