from datasets.encoder import get_codec
from model import GPT2, load_weight
from utils.utils import *
from utils.sample import *
import torch
import ipdb
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

titles = ["Muslims BUSTED: They Stole Millions In Gov’t Benefits",
"30 Civilians Die In US Airstrike Called ‘To Protect US and Afghan troops’",
"Clinton's Blitzkrieg Campaign: the Savage Politics of the Oligarchs",
"Watch: Rigged Voting Machine Will Not Allow Vote For Trump/Pence… “Stuck” On Clinton/Kaine",
"Stopping Hillary’s Coming War on Syria",
"FBI Believe Clinton Foundation Case Moving Towards ‘Likely an Indictment’",
"American Nightmare: Tyranny Under the FBI, an American Reality (Part III)",
"Girl Soldiers: Forgotten Casualties of War",
"The Last Confirmation Bias Test of This Election",
"Chinese may become Russia’s second largest ethnic population by 2050 | Russia & India Report",
"NEW WIKILEAK : Top Clinton Operative Believes “BLACK VOTERS ARE STUPID” – TruthFeed",
"“I feel like these Topshop models are sick of me apologising, but also maybe willing to hear me out”",
"Russian families with children in Europe have no way back to Russia",
"SUBHUMAN SAVAGES: ISIS shoots helpless disabled girl for delaying their convoy",
"Comment on Looking For Proof Of Evolution? You Can Find It On Your Own Body by spankthebank",
"See Dems accept foreign cash to disrupt Trump rallies",
"SNOWFLAKE: Colbert Didn’t Want To Write Jokes About Trump Winning | Daily Wire",
"While The FBI Investigates Hillary- WikiLeaks Swoops In To Kick Her While She’s Down",
"Newborn Baby Kidnapped from Alabama Hospital After Parents Decline Birth Certificate and SSN",
"‘Climate Emergency': North Pole Sees Record Temps, Melting Ice Despite Arctic Winter",
"Billionaire Globalist Soros Exposed as Hidden Hand Behind Trump Protests — Provoking US ‘Color Revolution’",
"Waking The Masses To The Climate Engineering Assault, Helpful Tools",
"Global markets turn red as Toblerone scandal unfolds",
"Joint Way Forward Deal Does Not Lead to Peace or Progress for Afghans",
"Comment on What Is Causing The Strange Trumpet Sounds In The Sky All Over The World? by MR.RANDY DOUGLAS MILLER",
"Assange: Clinton Campaign Tried To Hack WikiLeaks",
"After Correctly Predicting GDP Would Beat Estimates, Former Soros Associate Now Says Gold Headed Above $2,000",
"Van Full Of Illegals Shows Up To Vote Clinton At SIX Polling Places, Still Think Voter Fraud Is A Myth? – The Resistance: The Last Line of Defense",
"How Intellectual Property Rules Help the Rich and Hurt the Poor",
"Cast Your Vote: What Was Most Significant in Shaping the 2016 Election?",
"Donald Trump Promised to Bring Edward Snowden Home",
"Life: Preparing For Change: This Woman Is Scrambling To Experience As Much Human Dignity As Possible Before The Trump Administration Takes Power",
"LOL! MAD magazine designed HILARIOUS movie poster based on #PodestaEmails for Hillary’s new flick",
"Re: Can’t make this up: Michael Moore is pissed that his anti-Trump movie might help elect Donald Trump",
"Armenia genocide concert in Istanbul cancelled after reports of Erdogan being invited",
"Russia or the Neocons: Who endangers American democracy? | OffGuardian",
"No One Tried to Assassinate Donald Trump … But Austyn Crites Shows Up in WikiLeaks 7 Times",
"US uses Tunisia as drone base for Libya operations - report",
"Seven World-Historical Achievements of the Iraq Invasion of 2003",
"Why Hillary Clinton Will Appoint Old World Nationalists to Cabinet Positions",
"Hillary is finished! — as the FBI reopens their case into her latest crimes (Video)",
"Wolf Richter: Layoffs at Alphabet Access to Hit 9%, Google Fiber to “Pause” Plans, CEO Leaves, as Alphabet Cracks Down on Costs",
"SWEDEN HELP WANTED: Activities Coordinator for bored illegal alien Muslim freeloaders and rapists",
"Wolf Blitzer Walks Into Middle Of Olive Garden Commercial To Announce Breaking Election Results - The Onion - America's Finest News Source",
"Evil Russian Propaganda From The Evil Russian Invaders     : Information",
"Huma Abedin Swore Under Oath She Had No Emails, She Lied",
"New Male Birth Control Method Tested - The Onion - America's Finest News Source",
"Movie About Ireland’s Win Over New Zealand Announced",
"NYT Admits Key Al Qaeda Role in Aleppo",
"McLaren to Release Second Part of Report on Russia Doping Abuse in December",
"Clinton Struggles To Contain WikiLeaks Damage As Voters Grow Weary Of The Constant Scandals",
"Badass Patriot Has MASSIVE Surprise For Thieves Who Stole His Trump Sign",
"Oi veh! — New Hitler gets elected",
"Trump WON 1/3 the 700 Counties that Voted for Obama",
"Leaked Audio Of Hillary Clinton Proposing ‘Rigging Election’",
"Knife Review: Kellam Hawk Is Traditional Puukko Design with Modern Materials | Survival",
"Podesta Brothers Kidnapped a 3yo British Girl in Portugal - Wikileaks",
"The Truth U.S. Government Does NOT Want You To Know About The War In Syria",
"Scandalous Video Footage From Anonymous Exposes Huma and Hillary",
"8 Catastrophic Signs Reveal Hillary’s Campaign Is IMPLODING",
"Look What Students FORCED To Do After Muslims FLOOD Germany",
"Why Can't Hillary Clinton Stop Telling Stupid Lies?",
"Are The Polls Rigged Against Trump? All Of These Wildly Divergent Surveys Cannot Possibly Be Correct",
"Trump Won’t Mention That Bush & Cheney Deleted 22 Million Emails",
"Obama Won’t Vote With America at the UN",
"ISIS Killing Former Soldiers in Areas Around Mosul",
"Scientists Redefine Hurricanes and Deny Mini Ice Age will Effect Earth",
"A Tear in the Fabric of America’s Political Theater",
"Italy’s earthquakes leave 15,000 homeless, ‘soul of the country’ damaged",
"President of South Korea Is a Puppet of Her Rasputin-like Shaman Fortuneteller",
"Intervention Urged As RTÉ Still Insisting Nathan Carter Is A Thing",
"Former DOJ Spokesman SLAMS ‘Inappropriate’ FBI Statement On Hillary Emails (TWEETS, VIDEOS)",
"Citigroup bank chose Obama’s 2008 cabinet – WikiLeaks",
"Libertarian Party is DEAD after Endorsing Hillary Clinton",
"James Okeefe LIVE RADIO Wed Oct 26 th at bottom of hour 2:30 PM EDT on the Texas man's Radio show",
"Foreign Policy Names Floated By Trump Transition Undermine Campaign Promises",
"California & Massachusetts Just Legalized Recreational Marijuana",
"This cigarette lighter also doubles as a great bit of branded advertising",
"Re: It Is Now Mathematically Impossible To Pay Off The U.S. National Debt",
"Apple Kindly Offer Full-Time Jobs To Remaining 1,500 Calais Refugee Children",
"‘We The People’ Against Tyranny: Seven Principles For Free Government",
"BREAKING: Police Catch Iowa Man Who Executed Two Officers – And He’s A Trump Deplorable | If You Only News",
"Chart Of The Day: Mind The Big Rigs—-Class 8 Truck Sales Down 46% Since Peak",
"Neighbors Add To Hillary Sign With Blunt Message Of Their Own Next Door",
"They Said What?!: Find Out What Mindy Kaling, Barack Obama, And Steven Spielberg Have To Say",
"'D**k-Waving Berlusconi Knockoff': Late-Night Host's Nasty Trump Attack Dismantled by Steven Crowder",
"Life: 6 People Who Came To My Garage Sale Who Seem To Be Sticking Around For Dinner",
"Is A War In The Making — A Third World War? Instigated By A Declining Imperial Power",
"Joe Giambrone on Hollywood’s Shameless & Underhanded Assault on Political Truth",
"An FBI Agent Just Admitted His Bosses Are Pro-Trump, Hate “Antichrist” Hillary",
"Venezuela’s Economic Crisis: Does It Mean That the Left Has Failed?",
"South China Sea: Australia considers joint naval patrols with Indonesia",
"Scientists Say Universe Is Part Of 4th Dimension Born From Black Hole",
"Can The Oligarchy Still Steal The Presidential Election? — Paul Craig Roberts",
"White House Petition To Remove ‘Soros-Owned Voting Machines’ Hits 100,000+ Signatures",
"Trump and the Power of Money | The Vineyard of the Saker",
"Podesta Emails: Memo Reveals Clinton Campaign’s Plan To Coordinate With DNC",
"Santa Ana to pay $100,000 to pot shop at center of controversial police raid",
"November 1: Daily Contrarian Reads",
"Putin Ready to Restore Relations With US"]

def setup():
    args = parse_args()
    config = parse_config(args)
    np.random.seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.manual_seed(config.seed)

    codec = get_codec()
    model = GPT2(config)
    model = load_weight(model, torch.load('gpt2-pytorch_model.bin', map_location=device))
    model = model.to(device)
    model.eval()
    if not os.path.exists('submit'):
        os.makedirs('submit')

    return codec, model, config

def main():
    codec, model, config = setup()
    from utils.sample import sample
    with open(os.path.join('submit', 'samples.txt'), 'w', encoding='utf-8') as f:
        for i in range(len(titles)):
            start_text = titles[i]
            start_text = codec.encode(start_text).to(device)
            text = sample(model, start_text, config, codec)
            text = codec.decode(text.tolist()[0])

            f.write('=' * 50 + " SAMPLE_{} ".format(i) + '=' * 50 + '\n')
            f.write(text + '\n')
            print('=' * 50 + " SAMPLE_{} ".format(i) + '=' * 50 + '\n')
            print("Prompt: " + titles[i])
            print(text)
    print("# Samples written to samples.txt.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
