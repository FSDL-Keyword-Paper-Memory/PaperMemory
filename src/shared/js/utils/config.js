/**
 * Prototypes
 */

Object.defineProperty(Array.prototype, "last", {
    value: function (i = 0) {
        return this.reverse()[i];
    },
});

Object.defineProperty(String.prototype, "capitalize", {
    value: function () {
        return this.charAt(0).toUpperCase() + this.slice(1);
    },
});

/**
 * Global variable & constants are stored in this file to be used by
 * other files such as functions.js, parsers.js, memory.js, popup.js
 */

var global = {};
if (typeof window !== "undefined") {
    global = window;
}

/**
 * The popup's global state to store data across functions
 */
global.state = {
    dataVersion: 0,
    memoryIsOpen: false,
    menuIsOpen: false,
    papers: {},
    papersList: [],
    paperTags: new Set(),
    pdfTitleFn: null,
    showFavorites: false,
    sortedPapers: [],
    sortKey: "",
    papersReady: false,
    menu: {},
    files: {},
    ignoreSources: {},
    lastRefresh: new Date(),
};

global.descendingSortKeys = [
    "addDate",
    "count",
    "lastOpenDate",
    "favoriteDate",
    "year",
];

/**
 * Shared configuration for the Tags' select2 inputs
 */
global.select2Options = {
    placeholder: "Tag paper",
    maximumSelectionLength: 5,
    allowClear: true,
    tags: true,
    tokenSeparators: [",", " "],
};

/**
 * The array of keys in the menu, i.e. options the user can dis/enable in the menu
 */
global.menuCheckNames = [
    "checkBib",
    "checkMd",
    "checkDownload",
    "checkPdfTitle",
    "checkFeedback",
    "checkDarkMode",
    "checkDirectOpen",
    "checkStore",
    "checkScirate",
    "checkOfficialRepos",
    "checkPreferPdf",
    "checkPdfOnly",
];
/**
 * Menu check names which should not default to true but to false
 */
global.menuCheckDefaultFalse = [
    "checkDarkMode",
    "checkDirectOpen",
    "checkStore",
    "checkScirate",
    "checkOfficialRepos",
    "checkPdfOnly",
];
/**
 * All keys to retrieve from the menu, the checkboxes + the custom pdf function
 */
global.menuStorageKeys = [...global.menuCheckNames, "pdfTitleFn"];

/**
 * Map of known data sources to the associated paper urls: pdf urls and web-pages urls.
 * IMPORTANT: paper page before pdf (see background script)
 * Notes:
 *  ijcai -> papers < 2015 will not be parsed due to website changes
 *           (open an issue if that's problematic)
 */
global.knownPaperPages = {
    acl: ["aclanthology.org/"],
    acm: ["dl.acm.org/doi/"],
    acs: ["pubs.acs.org/doi/"],
    arxiv: ["arxiv.org/abs/", "arxiv.org/pdf/", "scirate.com/arxiv/"],
    biorxiv: ["biorxiv.org/content"],
    cvf: ["openaccess.thecvf.com/content"],
    ijcai: [(url) => /ijcai\.org\/proceedings\/\d{4}\/\d+/gi.test(url)],
    ieee: [
        "ieeexplore.ieee.org/document/",
        "ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=",
    ],
    iop: ["iopscience.iop.org/article/"],
    jmlr: [(url) => url.includes("jmlr.org/papers/v") && !url.endsWith("/")],
    nature: ["nature.com/articles/"],
    neurips: ["neurips.cc/paper/", "nips.cc/paper/"],
    openreview: ["openreview.net/forum", "openreview.net/pdf"],
    pmc: ["ncbi.nlm.nih.gov/pmc/articles/PMC"],
    pmlr: ["proceedings.mlr.press/"],
    pnas: ["pnas.org/content/", "pnas.org/doi/"],
};

global.sourcesNames = {
    acl: "Association for Computational Linguistics (ACL)",
    acm: "Association for Computing Machinery (ACM)",
    acs: "American Chemical Society (ACS)",
    arxiv: "ArXiv",
    biorxiv: "BioRxiv",
    cvf: "Computer Vision Foundation (CVF)",
    ijcai: "International Joint Conferences on Artificial Intelligence (IJCAI)",
    iop: "Institute Of Physics (IOP)",
    jmlr: "Journal of Machine Learning Research (JMLR)",
    nature: "Nature",
    neurips: "NeurIPS",
    openreview: "OpenReview",
    pmc: "PubMed Central",
    pmlr: "Proceedings of Machine Learning Research (PMLR)",
    pnas: "Proceedings of the National Academy of Sciences (PNAS)",
};

global.overrideORConfs = {
    "robot-learning": "CoRL",
    ijcai: "IJCAI",
};
global.overridePMLRConfs = {
    "Conference on Learning Theory": "CoLT",
    "International Conference on Machine Learning": "ICML",
    "Conference on Uncertainty in Artificial Intelligence": "UAI",
    "Conference on Robot Learning": "CoRL",
    "International Conference on Artificial Intelligence and Statistics": "AISTATS",
    "International Conference on Algorithmic Learning Theory": "ALT",
};
global.overrideDBLPVenues = {
    "J. Mach. Learn. Res.": "JMLR",
};

/**
 * Minimal Levenshtein distance between two paper titles for those to be merged
 */
global.fuzzyTitleMatchMinDist = 4;

global.defaultTitleFunctionCode = `
(paper) => {\n
    const title = paper.title.replaceAll("\\n", '');\n
    const id = paper.id;\n
    let name = \`\${title} - \${id}\`;\n
    name = name.replaceAll(":", " ").replace(/\\s\\s+/g, " ");\n
    return name\n};`;
global.storeReadme = `
/!\\ Warning: This folder has been created automatically by your PaperMemory browser extension.
/!\\ It has to stay in your downloads for PaperMemory to be able to access your papers.
/!\\ To be able to open files from this folder instead of re-downloading them, PaperMemory will match their titles and downloaded urls.
/!\\ If you change the default title function in the Advanced Options and do not include a paper's title in the file name, PaperMemory may not be able to open the file and will instead open the pdf url.
`;
/**
 * English words to ignore when creating an arxiv paper's BibTex key.
 */
global.englishStopWords = new Set([
    [
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "a",
        "an",
        "the",
        "and",
        "but",
        "if",
        "or",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "should",
        "now",
    ],
]);

// https://www.openacessjournal.com/journal-abbreviation-list
// Used for dblp matching
// adapted with common initials as JMLR
global.journalAbbreviations = {
    "infect dis poverty": "Infectious diseases of poverty",
    "emerg microbes infect": "Emerging microbes and infections",
    "front microbiol": "Frontiers in microbiology",
    "front cell infect microbiol": "Frontiers in Cellular and Infection Microbiology",
    "chin j acad radiol": "Chinese journal of academic radiology",
    "clin cosmet investig dermatol":
        "Clinical Cosmetic and Investigational Dermatology",
    "infect dis obstet gynecol": "Infectious Diseases in Obstetrics and Gynecology",
    "bmc dermatol": "BMC dermatology",
    "j invest dermatol": "Journal of Investigative Dermatology",
    "exp dermatol": "Experimental Dermatology",
    "br j dermatol": "British Journal of Dermatology",
    "j diabetes res": "Journal of Diabetes Research",
    "arch dermatol res": "Archives of Dermatological Research",
    "j am acad dermatol": "Journal of the American Academy of Dermatology",
    lancet: "The Lancet",
    "clin dermatol": "Clinics in Dermatology",
    "indian j obstet gynecol res":
        "Indian Journal of Obstetrics and Gynecology Research",
    "indian j med res": "Indian Journal of Medical Research",
    "am j clin dermatol": "American Journal of Clinical Dermatology",
    "j dermatol sci": "Journal of Dermatological Science",
    "indian j clin exp dermatol":
        "Indian Journal of Clinical and Experimental Dermatology",
    "panacea j med sci": "Panacea Journal of Medical Sciences",
    "indian j med sci": "Indian Journal of Medical Sciences",
    "clin exp dermatol": "Clinical and Experimental Dermatology",
    "diabetol int": "Diabetology international",
    "indian j community med": "Indian Journal of Community Medicine",
    "clin dermatol rev": "Clinical Dermatology Review",
    "indian dermatol online j": "Indian Dermatology Online Journal",
    "mvp j med sci": "MVP Journal of Medical Sciences",
    "indian j dermatol": "Indian Journal of Dermatology",
    "am j nurs": "American journal of nursing",
    "indian j dermatopathol diagn dermatol":
        "Indian Journal of Dermatopathology and Diagnostic Dermatology",
    "j neurosci": "Journal of neuroscience",
    "j food sci": "Journal of Food Science",
    "cardiovasc diabetol": "Cardiovascular Diabetology",
    "j med case rep": "Journal of medical case reports",
    "clin diabetes endocrinol": "Clinical Diabetes and Endocrinology",
    "j pediatr neurosci": "Journal of Pediatric Neurosciences",
    "j diabetes investig": "Journal of diabetes investigation",
    "world j diabetes": "World journal of diabetes",
    "j public health": "Journal of Public Health",
    "int j nanomed": "International Journal of Nanomedicine",
    "nutr diabetes": "Nutrition and Diabetes",
    "mol cancer": "Molecular Cancer",
    "indian j psychiatry": "Indian Journal of Psychiatry",
    "indian j microbiol res": "Indian Journal of Microbiology Research",
    "indian j orthop surg": "Indian Journal Of Orthopaedics Surgery",
    "indian j microbiol": "Indian journal of microbiology",
    "afr j paediatr surg": "African Journal of Paediatric Surgery",
    "ann pediatr card": "Annals of Pediatric Cardiology",
    "int j curr microbiol appl sci":
        "International Journal of Current Microbiology and Applied Sciences",
    "dermatol sin": "Dermatologica Sinica",
    "am j surg": "The American Journal of Surgery",
    "j am coll surg": "Journal of the American College of Surgeons",
    "indian j anaesth": "Indian Journal of Anaesthesia",
    "gen surg j": "General Surgery Journals",
    "indian j nephrol": "Indian Journal of Nephrology",
    "int j surg": "International Journal of Surgery",
    "indian j ophthalmol": "Indian Journal of Ophthalmology",
    "surg res pract": "Surgery Research and Practice",
    "indian j orthop": "Indian Journal of Orthopaedics",
    "clin surg": "Clinics in Surgery",
    "can j infect dis med microbiol":
        "Canadian Journal of Infectious Diseases and Medical Microbiology",
    "s afr j obstet gynaecol": "South African Journal of Obstetrics and Gynaecology",
    "int j nephrol renovascular dis":
        "International Journal of Nephrology and Renovascular Disease",
    "nepal j obstet gynecol": "Nepal Journal of Obstetrics and Gynaecology",
    "indian j clin exp ophthalmol":
        "Indian Journal Of Clinical And Experimental Ophthalmology",
    "indian j clin anaesth": "Indian Journal Of Clinical Anaesthesia",
    "indian j pathol oncol": "Indian Journal Of Pathology And Oncology",
    "indian j comm health": "Indian Journal of Community Health",
    "j pediatr crit care": "Journal Of Pediatric Critical Care",
    "j pathol clin res": "Journal of pathology Clinical research",
    "int j mycobacteriol": "International journal of mycobacteriology",
    "int j res dermatol": "International Journal of Research in Dermatology",
    "j am soc cytopathol": "Journal of the American Society of Cytopathology",
    "int j trop dis health": "International Journal of tropical disease and Health",
    "j emerg med": "The Journal of Emergency Medicine",
    "int j med paediatr oncol":
        "IP International Journal of Medical Paediatrics and Oncology",
    "j spine surg": "Journal of Spinal Surgery",
    "indian j pediatr": "Indian Journal of Pediatrics",
    "int j anat radiol surg": "International Journal of Anatomy Radiology and Surgery",
    "indian j public health": "indian journal of public health",
    "int j dermatopathol surg": "International Journal of Dermatopathology and Surgery",
    "j cancer tumor int": "Journal of Cancer and Tumor International",
    "anil aggrawals internet j forensic med toxicol":
        "Anil Aggrawals Internet Journal of Forensic Medicine and Toxicology",
    "acad pathol": "Academic Pathology",
    "eur j plant pathol": "European journal of plant pathology",
    "acad pediatr": "Academic Pediatrics",
    "int j biol macromol": "International Journal of Biological Macromolecules",
    "donald sch j ultrasound obstetrics gynecology":
        "Donald School Journal of Ultrasound in Obstetrics and Gynecology",
    "indian j med forensic med toxicol":
        "Indian Journal of Forensic Medicine and Toxicology",
    "j pathol inform": "Journal of Pathology Informatics",
    "indian j comp microbiol immunol infect dis":
        "Indian Journal of Comparative Microbiology Immunology and Infectious Diseases",
    "int j res orthop": "International Journal Of Research In Orthopaedics",
    "biomed res int": "Biomed research international",
    "orthop res rev": "Orthopedic Research and Reviews",
    "int j orthop rheumatol": "IP International Journal of Orthopaedic Rheumatology",
    "indian pediatr": "Indian Pediatrics",
    "int j pediatr": "International Journal of Pediatrics",
    "indian j pain": "Indian Journal of Pain",
    "indian j public health res dev":
        "Indian Journal of Public Health Research And Development",
    "j pure appl microbiol": "Journal of Pure and Applied Microbiology",
    "int j contemp med res": "International Journal of Contemporary Medical Research",
    "taiwan j psychiatry": "Taiwanese Society of Psychiatry",
    "int j indian psychol": "The International Journal of Indian Psychology",
    "int j anesth anesth": "International Journal of Anesthetics and Anesthesiology",
    "indian j child health": "Indian Journal of Child Health",
    "j clin anesth": "Journal of Clinical Anesthesia",
    "lancet child adolesc health": "The lancet child and adolescent health",
    "saudi j anaesth": "Saudi Journal of Anaesthesia",
    "j pediatr": "The Journal of Pediatrics",
    "j anaesthesiol clin pharmacol": "Journal of anaesthesiology clinical pharmacology",
    "j gynecol surg": "Journal of Gynecologic Surgery",
    "aesthetic surg j": "Aesthetic Surgery Journal",
    "br j anaesth": "British Journal of Anaesthesia",
    "j anesth clin res": "Journal of Anesthesia and Clinical Research",
    "best pract res clin anaesthesiol":
        "Best Practice and Research in Clinical Anaesthesiology",
    "j ophthalmol": "Journal of Ophthalmology",
    "ophthalmol retina": "Ophthalmology Retina",
    "turk j ophthalmol": "Turkish journal of ophthalmology",
    "indian j pathol microbiol": "Indian Journal of Pathology and Microbiology",
    "ann gastroenterol dig disord":
        "Annals of Gastroenterology and Digestive Disorders",
    "j can assoc gastroenterol":
        "Journal of the Canadian Association of Gastroenterology",
    "j clin gynaecol obstet": "Journal of Clinical Gynecology and Obstetrics",
    "ann saudi med": "Annals of Saudi Medicine",
    "j adv res": "Journal of advanced research",
    "j bone miner res": "Journal of bone and mineral research",
    "j bone jt surg": "The Journal of Bone and Joint Surgery",
    "curr opin gynecol obstet": "Current Opinion in Gynecology and Obstetrics",
    "indian j anat": "Indian Journal of Anatomy",
    "integr gynecol obstet j": "Integrative gynecology and obstetrics journal",
    "indian j clin anat physiol": "Indian Journal of Clinical Anatomy and Physiology",
    "ann anat": "Annals of Anatomy",
    "j orthop educ": "Journal of orthopedic education",
    "j ayurveda integr med": "Journal of Ayurveda and integrative medicine",
    ayu: "AYU An International Quarterly Journal of Research in Ayurveda",
    "am j pathol": "American Journal of Pathology",
    "anal cell pathol": "Analytical Cellular Pathology",
    "int j ayurveda pharma res":
        "International Journal of Ayurveda and Pharma Research",
    ayushdhara: "Ayushdhara",
    "n engl j med": "The New England Journal of Medicine",
    "int j pathol clin res": "International Journal of Pathology and Clinical Research",
    "ann med physiol": "Annals of Medical Physiology",
    "case rep orthop": "Case Reports in Orthopedics",
    "j orthop case rep": "Journal of Orthopaedic Case Reports",
    "int j womens health": "International Journal of Womens Health",
    "indones j soc obstet gynecol":
        "Indonesian Journal of Social Obstetrics and Gynecology",
    "am j obstet gynecol": "American Journal of Obstetrics and Gynecology",
    "acad forensic pathol": "Academic forensic pathology",
    "acta orthop": "Acta Orthopaedica",
    "aims microbiol": "AIMS microbiology",
    "anatol j cardiol": "Anatolian Journal of Cardiology",
    "indian j pathol res pract": "indian journal of pathology research and practice",
    "can j cardiol": "Canadian journal of cardiology",
    "cardiol res": "Cardiology Research",
    "j paediatr nurs sci": "IP Journal of Paediatrics and Nursing Science",
    "j diagn pathol oncol": "IP Journal of Diagnostic Pathology and Oncology",
    "j orthop": "Journal of Orthopaedics",
    "am j surg pathol": "The american journal of surgical pathology",
    "j pathol": "The Journal of pathology",
    "dev psychopathol": "Development and Psychopathology",
    "mod pathol": "Modern pathology",
    "case rep rheumatol": "Case Reports in Rheumatology",
    "brain pathol": "Brain pathology",
    "am j clin pathol": "American Journal of Clinical Pathology",
    "j clin pathol": "Journal of clinical pathology",
    "j public health res": "Journal of public health research",
    "am j public health": "American Journal of Public Health",
    "bmc public health": "BMC public health",
    "public health rep": "Public Health Reports",
    "eur j public health": "European Journal of Public Health",
    "j microsc ultrastruct": "Journal of Microscopy and Ultrastructure",
    "nat microbiol": "Nature microbiology",
    "jama ophthalmol": "JAMA Ophthalmology",
    "int j med sci clin invent":
        "International Journal of Medical Science and Clinical invention",
    "am j ophthalmol": "American Journal of Ophthalmology",
    "br j ophthalmol": "British Journal of Ophthalmology",
    "surv ophthalmol": "Survey of Ophthalmology",
    "pan asian j obstet gynecol": "Pan Asian Journal of Obstetrics and Gynecology",
    "am j psychiatry": "The American Journal of Psychiatry",
    "gen psychiatr": "General psychiatry",
    "telangana j psychiatry": "Telangana Journal of Psychiatry",
    "iran j neurol": "Iranian journal of neurology",
    "world j surg": "World Journal of Surgery",
    "j otorhinolaryngol allied sci":
        "IP Journal of Otorhinolaryngology and Allied Science",
    "ann natl acad med sci": "Annals of National Academy of Medical Sciences",
    "blde univ j health sciences": "Blde University Journal of Health Sciences",
    "arch med health sci": "Archives of Medicine and Health Sciences",
    "jmir pediatr parent": "JMIR pediatrics and parenting",
    "j pediatr pediatr med": "Journal of pediatrics and pediatric medicine",
    "pediatr qual saf": "Pediatric quality and safety",
    "tuberc res treat": "Tuberculosis Research and Treatment",
    "compr child adolesc nurs": "Comprehensive child and adolescent nursing",
    "curr treat options pediatr": "Current treatment options in pediatrics",
    "j diabetol": "Journal of Diabetology",
    "j pediatr neonatal care": "Journal of pediatrics and neonatal care",
    "int j contemp pediatr": "International journal of contemporary pediatrics",
    "child neurol open": "Child neurology open",
    "mol cell pediatr": "Molecular and cellular pediatrics",
    "int j ophthalmol": "International Journal of Ophthalmology",
    "j dent spec": "Journal of Dental Specialities",
    "j dent": "Journal of Dentistry",
    "jpn dent sci rev": "Japanese Dental Science Review",
    "indian j dent res": "Indian Journal of Dental Research",
    "indian j dent": "Indian Journal of Dentistry",
    "indian j multidiscip dent": "Indian Journal of Multidisciplinary Dentistry",
    "br dent j": "British Dental Journal",
    "j am dent assoc": "Journal of the american dental association",
    "ann maxillofac surg": "Annals of Maxillofacial Surgery",
    "dent res j": "Isfahan University of Medical Sciences",
    "int dent j stud res": "International Dental Journal of Student Research",
    "j contemp dent pract": "The Journal of Contemporary Dental Practice",
    "j endod": "Journal of Endodontics",
    "apos trends orthod": "APOS Trends in Orthodontics",
    "indian j conserv endod": "IP Indian Journal of Conservative and Endodontics",
    "int j maxillofac imaging": "IP International Journal of Maxillofacial Imaging",
    "int j oral health dent": "International Journal of Oral Health Dentistry",
    "j islamabad med dent college": "Journal of Islamabad Medical and Dental College",
    "world j dentistry": "World Journal of Dentistry",
    "j clin periodontol": "Journal of Clinical Periodontology",
    "j conserv dent": "Journal of conservative dentistry",
    "case rep dent": "Case Reports in Dentistry",
    "oral health dent manag": "Oral health and dental management",
    "int j appl dent sci": "International Journal of Applied Dental Sciences",
    "ann dent sci oral biol": "Annals of Dental Science and Oral Biology",
    "j dent oral biol": "Journal of dentistry and oral biology",
    "j oral biol craniofac res": "Journal of oral biology and craniofacial research",
    "j oral biosci": "Journal of oral biosciences",
    "j oral microbiol": "Journal of Oral Microbiology",
    "iran endod j": "Iranian Endodontic Journal",
    "dental press j orthod": "Dental Press Journal of Orthodontics",
    "j orofac sci": "Journal of Orofacial Sciences",
    "contemp clin dent": "Contemporary Clinical Dentistry",
    "clin oral implants res": "Clinical Oral Implants Research",
    "j restor dent": "Journal of Restorative Dentistry",
    "j esthet restor dent": "Journal of Esthetic and Restorative Dentistry",
    "restor dent endod": "Restorative Dentistry and Endodontics",
    "int j prosthodont endod":
        "International Journal of Prosthodontics and Restorative Dentistry",
    "ann prosthodont restor dent": "Annals of Prosthodontics and Restorative Dentistry",
    "indian j dent sci": "Indian Journal of Dental Sciences",
    "j indian soc periodontol": "Journal of Indian Society of Periodontology",
    "indian j dent educ": "Indian Journal of Dental Education",
    "sys rev pharm": "Systematic Reviews in Pharmacy",
    "indian j pharm pharmacol": "Indian Journal Of Pharmacy And Pharmacology",
    "indian j pharm sci": "Indian journal of pharmaceutical sciences",
    "int j pharm chem anal":
        "International Journal Of Pharmaceutical Chemistry And Analysis",
    "ann afr med": "Annals of African Medicine",
    "indian j pharmacol": "Indian Journal of Pharmacology",
    "indian j psychol med": "Indian Journal of Psychological Medicine",
    "j pharm biomed anal": "Journal of pharmaceutical and biomedical analysis",
    "j young pharm": "Journal of Young Pharmacists",
    "int j pharm pharm sci":
        "International Journal of Pharmacy and Pharmaceutical Sciences",
    "asian j pharm clin res":
        "The Asian Journal of Pharmaceutical and clinical research",
    "j oncol pract": "Journal of Oncology Practice",
    "j forensic leg med": "Journal of Forensic and Legal Medicine",
    "j forensic med": "Journal of Forensic Medicine",
    "int j forensic med toxicol sci":
        "IP International Journal of Forensic Medicine and Toxicological Sciences",
    "am j forensic med pathol": "American journal of forensic medicine and pathology",
    "int j med toxicol forensic med":
        "International Journal of Medical Toxicology and Forensic Medicine",
    "bangladesh j pharmacol": "Bangladesh Journal of Pharmacology",
    "j pharmacol pharmacother": "Journal of Pharmacology and Pharmacotherapeutics",
    "phcog rev": "Pharmacognosy Reviews",
    "j med pharm allied sci": "Journal of medical pharmaceutical and allied sciences",
    "j pharm biol sci": "Journal Of Pharmaceutical And Biological Sciences",
    "pharm chem j": "Pharmaceutical Chemistry Journal",
    "int j pharm pharm res":
        "International Journal of Pharmacy and Pharmaceutical Research",
    "am j med": "The american journal of medicine",
    "j am coll clin pharm": "Journal of the American College of Clinical Pharmacy",
    "pharm technol hosp pharm": "Pharmaceutical technology in hospital pharmacy",
    "sustain chem pharm": "Sustainable chemistry and pharmacy",
    "asian j pharm pharmacol": "Asian journal of pharmacy and pharmacology",
    "j pharm pharm": "Journal of pharmacy and pharmaceutics",
    "j pharm pharm sci": "Journal of Pharmacy and Pharmaceutical Sciences",
    "pharm pharmacol int j": "Pharmacy and pharmacology international journal",
    "j manag care spec pharm": "Journal of managed care and specialty pharmacy",
    "soj pharm pharm sci": "SOJ pharmacy and pharmaceutical sciences",
    "j pharm pharmacogn res": "Journal of pharmacy and pharmacognosy research",
    "j pharm policy pract": "Journal of pharmaceutical policy and practice",
    "world j pharm pharm sci": "World journal of pharmacy and pharmaceutical sciences",
    "j res pharm pract": "Journal of research in pharmacy practice",
    "pharm pat anal": "Pharmaceutical patent analyst",
    "biomed pharmacol j": "Biomedical and Pharmacology Journal",
    "acta med marisiensis": "Acta medica marisiensis",
    "int j clin pharm": "International journal of clinical pharmacy",
    "int j pharma bio sci": "International Journal of Pharma and Bio Sciences",
    "natl j physiol pharm pharmacol":
        "National journal of physiology pharmacy and pharmacology",
    "arch pharm pract": "Archives of pharmacy practice",
    "int rev comput softw": "International Review on Computers and Software",
    softwarex: "SoftwareX",
    "j big data": "Journal of Big Data",
    "comput struct biotechnol j": "Computational and Structural Biotechnology Journal",
    "j mach learn res": "JMLR",
    "peerj comput sci": "PeerJ Computer Science",
    "j comput des eng": "Journal of Computational Design and Engineering",
    "j ambient intell humaniz comput":
        "Journal of ambient intelligence and humanized computing",
    "int j artif intell": "International Journal of Artificial Intelligence",
    "j theor appl comput sci": "Journal of theoretical and applied computer science",
    "acm comput surv": "ACM Computing Surveys",
    "int j adv comput sci appl":
        "International Journal of Advanced Computer Science and Applications",
    "int j mach learn comput":
        "International journal of machine learning and computing",
    "procedia comput sci": "Procedia computer science",
    "npj quantum inf": "NPJ Quantum Information",
    "int j comput intell syst":
        "International Journal of Computational Intelligence Systems",
    "ieee trans pattern anal mach intell":
        "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    "ieee trans comput": "IEEE Transactions on Computers",
    "j acm": "Journal of the ACM",
    "ieee commun surv tutor": "IEEE Communications Surveys and Tutorials",
    "int j data sci anal": "International Journal of Data Science and Analytics",
    "int j comput technol": "International journal of computers and technology",
    "big data soc": "Big Data and Society",
    "adv hum comput interact": "Advances in Human Computer Interaction",
    "australas j educ technol": "Australasian Journal of Educational Technology",
    "indian j law hum behav": "Indian Journal of Law and Human Behaviour",
    "int j soc welf manage": "International Journal Of Social Welfare And Management",
    "j nanotechnol": "Journal of nanotechnology",
    "vietnam j sci technol": "Vietnam journal of science and technology",
    "j mater sci": "Journal of Materials Science",
    "mater res express": "Materials research express",
    "j mater sci mater electron":
        "Journal of Materials Science Materials in Electronics",
    "nat mater": "Nature Materials",
    "adv mater": "Advanced Materials",
    "mater today": "Materials Today",
    "annu rev mater res": "Annual Review of Materials Research",
    "acs appl mater interfaces": "ACS Applied Materials and Interfaces",
    "adv mat res": "Advanced Materials Research",
    "adv mater interfaces": "Advanced materials interfaces",
    "iete j res": "IETE journal of research",
    "j environ chem eng": "Journal of Environmental Chemical Engineering",
    "int j eng adv technol":
        "International Journal of Engineering and Advanced Technology",
    "arab j sci eng": "Arabian journal for science and engineering",
    "j build perform simul": "International Journal of Building Performance Simulation",
    "int j eng res appl":
        "International journal of engineering research and applications",
    "eng j": "Engineering Journal",
    "j eng res": "The Journal of Engineering Research",
    "int j eng res": "International Journal of Engineering Research",
    "int j recent technol eng":
        "International Journal of Recent Technology and Engineering",
    "eng sci technol int j":
        "Engineering science and technology an international journal",
    "adbu j eng technol": "ADBU Journal of Engineering Technology",
    "adv electron forum": "Advanced Engineering Forum",
    "curr eng j": "Current Biochemical Engineering",
    "int j adv eng sci appl math":
        "International Journal of Advances in Engineering Sciences and Applied Mathematics",
    "int res j eng technol":
        "International Research Journal of Engineering and Technology",
    "j constr manag": "Journal of Construction Management",
    "polym eng sci": "Polymer engineering and science",
    "adv phys": "Advances in Physics",
    "int j electrochem": "International Journal of Electrochemistry",
    "bulg j phys": "Bulgarian Journal of Physics",
    "adv condens matter phys": "Advances in Condensed Matter Physics",
    "open access j med aromat plants":
        "Open Access Journal of Medicinal and Aromatic Plants",
    "hepatol int": "Hepatology International",
    phytomedicine:
        "Phytomedicine International Journal of Phytotherapy and Phytopharmacology",
    "indian j gastroenterol": "Indian Journal of Gastroenterology",
    "antioxid redox signal": "Antioxidants and Redox Signaling",
    "j chem educ": "Journal of Chemical Education",
    "j org chem": "Journal of Organic Chemistry",
    "j biol chem": "Journal of biological chemistry",
    "bmc chem": "BMC chemistry",
    "indian j adv chem sci": "Indian Journal of Advances in Chemical Science",
    "asian j chem": "Asian Journal of Chemistry",
    "j chem sci": "Journal of Chemical Sciences",
    "res j chem sci": "Research Journal of Chemical Sciences",
    "j chem": "Journal of Chemistry",
    "orient j chem": "Oriental Journal of Chemistry",
    "new j chem": "New journal of chemistry",
    "anal chem lett": "Analytical Chemistry Letters",
    "am j transplant": "American Journal of Transplantation",
    "indian j surg": "Indian journal of surgery",
    "jama surg": "JAMA Surgery",
    "ind j agric bus": "Indian Journal of Agriculture Business",
    "acta agric scand a anim sci":
        "Acta Agriculturae Scandinavica Section A Animal Science",
    "ind j agric biochem": "Indian Journal of Agricultural Biochemistry",
    "indian j agric res": "Indian Journal of Agricultural Research",
    "indian j agron": "Indian Journal of Agronomy",
    "physiol mol plant pathol": "Physiological and Molecular Plant Pathology",
    "indian j genet plant breed": "Indian Journal of Genetics and Plant Breeding",
    "indian j agr sci": "indian journal of agricultural sciences",
    "j agric econ": "journal of agricultural economics",
    "j agron indones": "Jurnal Agronomi Indonesia",
    "int j agric sci": "International Journal of Agricultural Sciences",
    "open agric j": "The Open Agriculture Journal",
    "acta fytotech zootech": "Acta Fytotechnica et Zootechnica",
    "j agric nat resour": "Journal of Agriculture and Natural Resources",
    "j nepal agric res counc": "Journal of Nepal Agricultural Research Council",
    "adv sci lett": "Advanced Science Letters",
    "adv agric": "Advances in Agriculture",
    "asian j agric dev": "Asian Journal of Agriculture and Development",
    "asian j agric food sci": "Asian journal of agriculture and food science",
    "int j agric innov res":
        "International journal of agriculture innovations and research",
    "j integr agric": "Journal of integrative agriculture",
    "agric human values": "Agriculture and human values",
    "front agric china": "Frontiers of Agriculture in China",
    "j exp biol agric sci": "Journal of Experimental Biology and Agricultural Sciences",
    "j community health manag": "The Journal of Community Health Management",
    "j patient saf risk manag": "Journal of patient safety and risk management",
    "perspect public manag gov": "Perspectives on public management and governance",
    "j manag res anal": "Journal Of Management Research And Analysis",
    "acad manag learn educ": "Academy of Management Learning and Education",
    "j bus res": "Journal of Business Research",
    "acad manage j": "Academy of Management Journal",
    "acad manage rev": "Academy of Management Review",
    "account organ soc": "Accounting Organizations and Society",
    "am econ rev": "The American Economic Review",
    "j int bus stud": "Journal of International Business Studies",
    "j int econ": "Journal of International Economics",
    "j manage": "Journal of management",
    "aims int j manag": "AIMS International Journal of Management",
    "int j bus manag": "International Journal of Business and Management",
    "eur manag j": "European Management Journal",
    "eur j manag": "European Journal of Management",
    "eur j int manag": "European Journal of International Management",
    "": "Advances in Statistical Climatology Meteorology and Oceanography",
    "int interdiscip res j": "International interdisciplinary research journal",
    "j anim sci": "Journal of Animal Science",
    "anim nutr technology": "Animal Nutrition and Feed Technology",
    "j anim res": "Journal of Animal Research",
    "anim prod sci": "Animal Production Science",
    "asian australas j anim sci": "Asian Australasian journal of animal sciences",
    "annu rev stat appl": "Annual review of statistics and its application",
    "ann math": "Annals of mathematics",
    "am stat": "The American Statistician",
    "aust n z j stat": "Australian and New Zealand journal of statistics",
    "j stat softw": "Journal of statistical software",
    "int j sci technol res":
        "International Journal of Scientific and Technology Research",
    "int j mod trends sci technol":
        "International Journal for Modern Trends in Science and Technology",
    "int j sci res eng dev":
        "International Journal of Scientific Research and Engineering Development",
    "indian j sci technol": "Indian Journal of Science and Technology",
    "indian j biotechnol": "Indian journal of biotechnology",
    "east asian sci technol soc": "East Asian science technology and society",
    "east asian sci technol med": "East Asian science technology and medicine",
    "sci technol adv mate": "Science and technology of advanced materials",
    "j mar sci technol": "Journal of Marine Science and Technology",
    "open access j sci technol": "Open Access Journal of Science and Technology",
    "sci technol stud": "Science and Technology Studies",
    "j soc sci": "Journal of Social Sciences",
    "mediterr j soc sci": "Mediterranean journal of social sciences",
    "j progress res soc sci": "Journal of progressive research in social sciences",
    rsf: "The Russell Sage Foundation journal of the social sciences",
    "j stud soc sci": "Journal of studies in social sciences",
    "j methods meas soc sci":
        "Journal of methods and measurement in the social sciences",
    "j rural soc sci": "Journal of rural social sciences",
    methodology:
        "European journal of research methods for the behavioral and social sciences",
    "pak j life soc sci": "Pakistan journal of life and social sciences",
    "am j islam soc sci": "The American journal of Islamic social sciences",
    "soc sci china": "Social sciences in China",
    "pak soc sci rev": "Pakistan social sciences review",
    "acta univ sapientiae econ bus":
        "Acta Universitatis Sapientiae Economics and Business",
    "adv soc work": "Advances in Social Work",
    "int j soc sci humanity stud":
        "International Journal of Social Sciences and Humanity Studies",
    "int j humanit soc sci": "International Journal of Humanities and Social Science",
    "am j bus": "American Journal of Business",
    "int j soc sci stud": "International Journal of Social Science Studies",
    "ann econ financ": "Annals of Financial Economics",
    "soc sci j": "The Social Science Journal",
    "asia pac j public adm": "Asia Pacific Journal of Public Administration",
    "am j sociol": "American Journal of Sociology",
    "asia pac j risk insur": "Asia Pacific Journal of Risk and Insurance",
    "chin j sociol": "Chinese journal of sociology",
    "aust intellect prop j": "Australian Intellectual Property Journal",
    "am j cult sociol": "American Journal of Cultural Sociology",
    "anthropol j food": "Anthropology of Food",
    "adv appl sociol": "Advances in applied sociology",
    "asian assoc open univ j": "Asian Association of Open Universities Journal",
    "j appl soc sci": "Journal of applied social science",
    "asian j bus ethics": "Asian Journal of Business Ethics",
    "adv j soc sci": "Advanced Journal of Social Science",
    "int j soc educ sci": "International Journal on Social and Education Sciences",
    "int j yoga": "International Journal of Yoga",
    "indian j anc med yoga": "Indian Journal of Ancient Medicine and Yoga",
    "j yoga phys ther": "Journal of yoga and physical therapy",
    "int j yoga therap": "International journal of yoga therapy",
    "nat biotechnol": "Nature Biotechnology",
    "trends biotechnol": "Trends in biotechnology",
    "biotechnol adv": "Biotechnology Advances",
    "j appl biol biotechnol": "Journal of Applied Biology and Biotechnology",
    "int j biotechnol": "International journal of biotechnology",
    "pak j bot": "Pakistan Journal of Botany",
    "ann bot": "Annals of Botany",
    "aquat bot": "Aquatic Botany",
    "aust j bot": "Australian Journal of Botany",
    "aust syst bot": "Australian Systematic Botany",
    "bot lett": "Botany letters",
    "curr protoc plant biol": "Current protocols in plant biology",
    "plant biotechnol j": "Plant Biotechnology Journal",
    "curr trends biotechnol pharm": "Current Trends in Biotechnology and Pharmacy",
    "j plant sci res": "Journal of Plant Science and Research",
    "int j glob sci res": "International Journal of Global Science Research",
    "j clean prod": "Journal of cleaner production",
    "nat commun": "Nature Communications",
    "j environ sci": "Journal of Environmental Sciences",
    "environ anal health toxicol": "Environmental analysis health and toxicology",
    "curr opin environ sci health":
        "Current Opinion in Environmental Science and Health",
    "appl ecol environ sci": "Applied Ecology and Environmental Sciences",
    "adv environ res": "Advances in Environment Research",
    "nat environ pollut technol": "Nature Environment and Pollution Technology",
    "j assoc environ resour econ":
        "Journal of the Association of Environmental and Resource Economists",
    "int j sci res": "International journal of scientific research",
    "adv appl math sci": "Advances and Applications in Mathematical Sciences",
    "aims mol sci": "AIMS Molecular Science",
    "sci rep": "Scientific Reports",
    "acta ger": "Acta Germanica",
    amphora: "Amphora",
    "antennae j nat vis culture": "Antennae The Journal of Nature in Visual Culture",
    "art market": "Arts and the Market",
    "archaeol int": "Archaeology International",
    "iafor j arts humanit": "IAFOR Journal of Arts and Humanities",
    "iis univ j arts": "IIS University Journal of Arts",
    "j lang discrimination": "Journal of Language and Discrimination",
    "int crit theor": "International Critical Thought",
    "int j comp lit translat stud":
        "International Journal of Comparative Literature and Translation Studies",
    "iafor j ethics religion philos": "IAFOR Journal of Ethics Religion and Philosophy",
    "between species": "Between the Species",
    "comp philos": "Comparative Philosophy",
    "j pop telev": "Journal of Popular Television",
    "art public sphere": "Art and the Public Sphere",
    "environ philos": "Environmental Philosophy",
    "focus ger stud": "Focus on German Studies",
    "ger foreign lang": "German as a Foreign Language",
    "black camera": "Black Camera",
    "life sci soc policy": "Life Sciences Society and Policy",
    "anc sci life": "Ancient Science of Life",
    "annals ayurvedic med": "Annals of Ayurvedic Medicine",
    "arab j math sci": "Arab Journal of Mathematical Sciences",
    "arab j math": "Arabian Journal of Mathematics",
    "albanian j math": "Albanian Journal of Mathematics",
    "indian drugs": "Indian Drugs",
    "int j clin biochem res":
        "International Journal of Clinical Biochemistry and Research",
    biochem: "Biochemistry",
    "contemp account res": "Contemporary accounting research",
    "j account res": "Journal of Accounting Research",
    "j account econ": "Journal of Accounting and Economics",
    "biosci biotechnol res asia": "Biosciences Biotechnology Research Asia",
    "j bank financ": "Journal of Banking and Finance",
    "int j bank account finance":
        "International Journal of Banking Accounting and Finance",
    "indian j bank financ": "Indian Journal of Finance and Banking",
    "int j cent bank": "International journal of central banking",
    "cell tissue bank": "Cell and Tissue Banking",
    "appl finance account": "Applied finance and accounting",
    "j finance": "Journal of Finance",
    "int j orthop sci": "International Journal of Orthopaedics Sciences",
    "food energy secur": "Food and Energy Security",
    "j food drug anal": "Journal of food and drug analysis",
    "j nutr sci": "Journal of nutritional science",
    "j wildl manage": "Journal of Wildlife Management",
    "j bus ethics": "Journal of Business Ethics",
    "j bus anal": "Journal of business analytics",
    "int j nurs stud": "International journal of nursing studies",
    "j nurs educ pract": "Journal of nursing education and practice",
    "adv emerg nurs j": "Advanced emergency nursing journal",
    "ans adv nurs sci": "Advances in Nursing Science",
    "j carcinog": "Journal of Carcinogenesis",
    "j pers soc psychol": "Journal of personality and social psychology",
    "j mol liq": "Journal of Molecular Liquids",
    "malar control elimin": "Malaria control and elimination",
    "malar j": "Malaria Journal",
    "conserv soc": "Conservation and Society",
};

global.art = {
    dolphin: `
            ;'-.
\`;-._        )  '---.._
    >  \`-.__.-'          \`'.__
    /_.-'-._         _,   ^ ---)
    \`       \`'------/_.'----\`\`\`

    `,
    shark: `
                (\`.
                 \\ \`.
                  )  \`._..---._
\\\`.       __...---\`         o  )
 \\ \`._,--'           ,    ___,'
  ) ,-._          \\  )   _,-'
 /,'    \`\`--.._____\\/--''

    `,
    bat: `
   /\\                 /\\
  / \\'._   (\\_/)   _.'/ \\
 /_.''._'--('.')--'_.''._\\
 | \\_ / \`;=/ " \\=;\` \\ _/ |
  \\/ \`\\__|\`\\___/\`|__/\`  \\/
   \`      \\(/|\\)/       \`
           " \` "

    `,
    cat: `
   |\\---/|
   | ,_, |
    \\_\`_/-..----.
 ___/ \`   ' ,""+ \\
(__...'   __\\    |\`.___.';
  (_,...'(_,.\`__)/'.....+

    `,
    dog: `
       /^-^\\
      / o o \\
     /   Y   \\
     V \\ v / V
       / - \\
      /    |
(    /     |
 ===/___) ||

    `,
    triceratops: `
                        . - ~ ~ ~ - .
      ..     _      .-~               ~-.
     //|     \\ \`..~                      \`.
    || |      }  }              /       \\  \\
(\\   \\\\ \\~^..'                 |         }  \\
 \\\`.-~  o      /       }       |        /    \\
 (__          |       /        |       /      \`.
  \`- - ~ ~ -._|      /_ - ~ ~ ^|      /- _      \`.
              |     /          |     /     ~-.     ~- _
              |_____|          |_____|         ~ - . _ _~_-_

    `,
    moose: `
 ___            ___
/   \\          /   \\
\\_   \\        /  __/
 _\\   \\      /  /__
 \\___  \\____/   __/
     \\_       _/
       | @ @  \\_
       |
     _/     /\\
    /o)  (o/\\ \\_
    \\_____/ /
      \\____/

    `,
    bear: `

     (()__(()
     /       \\
    ( /    \\  \\
     \\ o o    /
     (_()_)__/ \\
    / _,==.____ \\
   (   |--|      )
   /\\_.|__|'-.__/\\_
  / (        /     \\
  \\  \\      (      /
   )  '._____)    /
(((____.--(((____/

    `,
};
