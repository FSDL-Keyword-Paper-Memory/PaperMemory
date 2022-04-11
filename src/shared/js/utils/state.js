/**
 * Function to initialize the app's state.
 *  1. load the papers from storage is the papers argument is undefined
 *  2. load and check the user's title function from storage
 *  3. apply migration on memory data
 *  4. if isContentScript is true, return
 *  5. set the memory display state attributes
 *     a. papersList
 *     b. set default sort key to lastOpenDate
 *     c. papersReady is now true
 *     d. set the menu from storage
 *  6. Sort the memory's papers
 *  7. Discover all the memory's tags
 *
 * @param {object} papers Memory object with papers to initialize the state with
 * @param {boolean} isContentScript Whether the call is from a content_script or the popup
 */
const initState = async (papers, isContentScript) => {
    const s = Date.now();
    if (typeof papers === "undefined") {
        papers = await getStorage("papers");
        info("Time to retrieve stored papers (s): " + (Date.now() - s) / 1000);
    }

    global.state.dataVersion = getManifestDataVersion();
    global.state.titleFunction = (await getTitleFunction()).titleFunction;

    weeklyBackup();

    const migration = await migrateData(papers, global.state.dataVersion);

    papers = migration.papers;
    global.state.papers = papers;
    global.state.menu = await getMenu();
    global.state.ignoreSources = (await getStorage("ignoreSources")) || {};

    if (isContentScript) {
        info("State initialization duration (s): " + (Date.now() - s) / 1000);
        return;
    }
    global.state.files = await matchAllFilesToPapers();
    global.state.papersList = Object.values(cleanPapers(papers));
    global.state.sortKey = "lastOpenDate";
    global.state.papersReady = true;

    sortMemory();
    makeTags();

    info("State initialization duration (s): " + (Date.now() - s) / 1000);
};

/**
 * Sample a paper from the memory. If `idx` is provided, the paper will
 * be the `idx`-th paper in the memory (in the list of keys).
 * Otherwise, a random paper will be drawn.
 * If no paper exists in the memory a dummy paper object is returned.
 *
 * @param {number} idx Optional index of the sample paper.
 * @returns {object} paper object to display in the options.
 */
const getExamplePaper = async (idx) => {
    // all papers
    const papers = await getStorage("papers");
    // filter out the data version
    const keys = Object.keys(papers)
        .filter((k) => k.indexOf("__") === -1)
        .reverse();
    // no idx provided, sample a random paper
    if (typeof idx === "undefined") {
        idx = getRandomInt(keys.length);
    }
    let paper = papers[keys[idx]];
    // there's no such paper (idx is wrong or memory is empty)
    if (typeof paper === "undefined") {
        paper = {
            title: "Dummy title",
            author: "Cool Author and Great Author and Complicated Name Àuthor",
            year: 2021,
            id: "NoneXiv-214324",
            bibtex: "@Nonesense{}",
            tags: ["t1", "t2"],
            note: "Thispaperdoesnotexist.com",
        };
    }
    return paper;
};

/**
 * Tries to parse the text input by the user to define the function that takes
 * a paper in order to create the custom page title / pdf filename.
 * If there is an error, it uses the built-in function from global.defaultTitleFunctionCode.
 * @param {string} code The string describing the code function.
 * @returns {function} Either the user's function if it runs without errors, or the built-in
 * formatting function
 */
const getTitleFunction = async (code = null) => {
    let titleFunction;
    if (!code) {
        // no code provided: load it from storage
        code = await getStorage("titleFunctionCode");
    }
    if (typeof code === "undefined") {
        // no code provided and no code in storage: use the built-in function code
        code = global.defaultTitleFunctionCode;
    }

    let errorMessage;

    try {
        // eval the code into a function
        titleFunction = eval(code);
    } catch (error) {
        // there was an error evaluating the code: use the built-in function
        errorMessage = `Error parsing the title function: ${error}`;
        log("Error parsing the title function. Function string then error:");
        log(code);
        log(error);
        titleFunction = eval(global.defaultTitleFunctionCode);
        code = global.defaultTitleFunctionCode;
    }

    try {
        // test the loaded title function on a paper
        const examplePaper = await getExamplePaper(0);
        const result = titleFunction(examplePaper);
        // assert the result is a string
        if (typeof result !== "string") {
            throw new Error(`Result ${result} is not a string`);
        }
    } catch (error) {
        // there was an error running the title function: use the built-in function
        errorMessage = `Error executing the title function: ${error}`;
        log("Error testing the user's title function. Function string then error:");
        log(code);
        log(error);
        titleFunction = eval(global.defaultTitleFunctionCode);
        code = global.defaultTitleFunctionCode;
    }

    return {
        titleFunction: titleFunction, // the function to use
        code: code.trim(), // the code to store in storage
        errorMessage: errorMessage, // potential error message
    };
};

/**
 * Uses the state-loaded title function to get the title of a paper.
 *
 * @param {string || object} paperOrId the paper for which to get the title
 * @returns {string} the title of the paper
 */
const stateTitleFunction = (paperOrId) => {
    let paper = paperOrId;
    if (typeof paperOrId === "string") {
        // paperOrId is an ID
        paper = global.state.papers[paperOrId];
        if (typeof paper === "undefined") {
            // no such paper
            log("Error in stateTitleFunction: unknown id", paperOrId);
            return "Unknown ID";
        }
    }
    let name;
    try {
        // try using the state-loaded title function
        name = global.state.titleFunction(paper);
    } catch (error) {
        // there was an error: use the built-in function
        log("Error in stateTitleFunction:", error);
        name = eval(global.defaultTitleFunctionCode)(paper);
    }
    // make sure there's no new line and no double spaces in the title
    return name.replaceAll("\n", " ").replace(/\s\s+/g, " ");
};

/**
 *  Adds a new paper to the memory or updates the counts and open dates of an existing paper.
 *
 * @param {string} url The url from which to parse a paper
 * @param {object} is The paper's source info
 * @param {object} checks The user's preferences
 * @returns
 */
const addOrUpdatePaper = async (url, is, menu) => {
    let paper, isNew, pwcUrl, pwcNote, pwcVenue;

    // Extract id from url
    const id = await parseIdFromUrl(url);
    log("id:", id);

    if (id && global.state.papers.hasOwnProperty(id)) {
        // Update paper if it exists
        global.state.papers = updatePaper(global.state.papers, id);
        paper = global.state.papers[id];
        isNew = false;
    } else {
        // Or create a new one if it does not
        paper = await makePaper(is, url, id);
        if (!paper) {
            return;
        }
        const existingId = null; // findFuzzyPaperMatch(paper); // not used yet.
        if (existingId) {
            // Currently the code never goes here

            // Update paper as already it exists
            info(
                `Found a fuzzy match for (${id}) ${paper.title}`,
                global.state.papers[existingId]
            );
            global.state.papers = updatePaper(global.state.papers, existingId);
            paper = global.state.papers[existingId];
            isNew = false;
        } else {
            // store the new paper in the global state
            global.state.papers[paper.id] = paper;
            // set isNew to True for the storage setter
            isNew = true;
        }
    }

    if (!paper.codeLink || !paper.venue) {
        try {
            const payload = {
                type: "papersWithCode",
                paper: paper,
                officialReposOnly: menu.checkOfficialRepos,
            };
            const pwc = await sendMessageToBackground(payload);

            pwcUrl = !paper.codeLink && pwc?.url;
            pwcNote = pwc?.note;
            pwcVenue = !paper.venue && pwc?.venue;

            if (pwcUrl) {
                console.log(
                    "Discovered a code repository from PapersWithCode:",
                    pwcUrl
                );
                paper.codeLink = pwcUrl;
                if (pwc.hasOwnProperty("note")) delete pwc.note;
                paper.code = pwc;
            }
        } catch (error) {
            log("Error trying to discover a code repository:");
            log(error);
        }
    }

    global.state.papers = await getStorage("papers");
    global.state.papers[paper.id] = paper;

    chrome.storage.local.set({ papers: global.state.papers }, async () => {
        let notifText;
        if (isNew || pwcUrl) {
            if (isNew) {
                // new paper

                info("Added '" + paper.title + "' to your Memory!");
                log("paper: ", paper);
                notifText = "Added to your Memory";
                if (pwcUrl) {
                    notifText +=
                        "<br/><div id='feedback-pwc'>(+ repo from PapersWithCode) </div>";
                }
                menu && menu.checkFeedback && feedback(notifText, paper);
            } else {
                // existing paper but new code repo

                notifText = "Found a code repository on PapersWithCode!";
                menu && menu.checkFeedback && feedback(notifText);
            }
        } else {
            info("Updated '" + paper.title + "' in your Memory");
        }

        if (!paper.venue && !pwcVenue) {
            log("[PapersWithCode] No publication found");
        }

        // anyway: try and update note with actual publication
        if (!paper.note || !paper.venue) {
            const { note, venue, bibtex } = await tryPreprintMatch(paper);
            if (note || venue || bibtex) {
                if (!paper.note) {
                    if (note) {
                        log("Updating preprint note to", note);
                        paper.note = note;
                    } else if (pwcNote) {
                        log("Updating preprint note to", pwcNote);
                        paper.note = pwcNote;
                    }
                }
                if (!paper.venue) {
                    if (venue) {
                        log("Updating preprint venue to", venue);
                        paper.venue = venue;
                    } else if (pwcVenue) {
                        log("Updating preprint venue to", pwcVenue);
                        paper.venue = pwcVenue;
                    }
                }
                if (bibtex) {
                    log("Updating preprint bibtex to", bibtex);
                    paper.bibtex = bibtex;
                }
                global.state.papers = await getStorage("papers");
                global.state.papers[paper.id] = paper;
                chrome.storage.local.set({ papers: global.state.papers });
            }
        }
    });

    return { paper, id };
};

/**
 * Parses a paper's id from a url.
 * Throws error if the url is not a paper source as defined per isPaper(url).
 *
 * @param {string} url The url to use in order to find a matching paper
 * @returns {string} The id of the paper found.
 */
const parseIdFromUrl = async (url) => {
    const is = await isPaper(url, true);
    if (is.arxiv) {
        const arxivId = url.match(/\d{4}\.\d{4,5}/g)[0];
        return `Arxiv-${arxivId}`;
    } else if (is.neurips) {
        const year = url.split("/paper/")[1].split("/")[0];
        const hash = url.split("/").last().split("-")[0].slice(0, 8);
        return `NeurIPS-${year}_${hash}`;
    } else if (is.cvf) {
        return parseCVFUrl(url).id;
    } else if (is.openreview) {
        const OR_id = url.match(/id=\w+/)[0].replace("id=", "");
        const paper = Object.values(cleanPapers(global.state.papers)).filter((p) => {
            return p.id.includes(OR_id);
        })[0];
        return paper && paper.id;
    } else if (is.biorxiv) {
        url = cleanBiorxivURL(url);
        let id = url.split("/").last();
        if (id.match(/v\d+$/)) {
            id = id.split("v")[0];
        }
        return `Biorxiv-${id}`;
    } else if (is.pmlr) {
        const key = url.split("/").last().split(".")[0];
        const year = "20" + key.match(/\d+/)[0];
        return `PMLR-${year}-${key}`;
    } else if (is.acl) {
        url = url.replace(".pdf", "");
        if (url.endsWith("/")) {
            url = url.slice(0, -1);
        }
        const key = url.split("/").last();
        const paper = Object.values(cleanPapers(global.state.papers)).filter((p) => {
            return p.source === "acl" && p.id.includes(key);
        })[0];
        return paper && paper.id;
    } else if (is.pnas) {
        url = url.replace(".full.pdf", "");
        const pid = url.endsWith("/")
            ? url.split("/").slice(-2)[0]
            : url.split("/").slice(-1)[0];

        const paper = Object.values(cleanPapers(global.state.papers)).filter((p) => {
            return p.source === "pnas" && p.id.includes(pid);
        })[0];
        return paper && paper.id;
    } else if (is.nature) {
        url = url.replace(".pdf", "").split("#")[0];
        const hash = url.split("/").last();
        return `Nature_${hash}`;
    } else if (is.acs) {
        url = url
            .replace("pubs.acs.org/doi/pdf/", "/doi/")
            .replace("pubs.acs.org/doi/abs/", "/doi/")
            .split("?")[0];
        const doi = url.split("/doi/")[1].replaceAll(".", "").replaceAll("/", "");
        return `ACS_${doi}`;
    } else if (is.iop) {
        url = url.split("#")[0].replace(/\/pdf$/, "");
        const doi = url.split("/article/")[1].replaceAll(".", "").replaceAll("/", "");
        return `IOPscience_${doi}`;
    } else if (is.jmlr) {
        if (url.endsWith(".pdf")) {
            url = url.split("/").slice(0, -1).join("/");
        }
        url = url.replace(".html", "");
        const jid = url.split("/").last();
        const year = `20${jid.match(/\d+/)[0]}`;
        return `JMLR-${year}_${jid}`;
    } else if (is.pmc) {
        const pmcid = url.match(/PMC\d+/g)[0].replace("PMC", "");
        const paper = Object.values(cleanPapers(global.state.papers)).filter((p) => {
            return p.source === "pmc" && p.id.includes(pmcid);
        })[0];
        return paper && paper.id;
    } else if (is.localFile) {
        return is.localFile;
    } else {
        throw new Error("unknown paper url");
    }
};

/**
 * A function to detect whether the current local file is known in the
 * user's memory by ensuring there exists a paper whose title is in the filename.
 * The loose comparison is made against lowercase and letter-only strings.
 *
 * @param {string} url The current url (potentially: local file)
 * @returns {string || boolean} false if no paper is found, the paper's id if found
 */
const isKnownLocalFile = (url) => {
    if (!url.startsWith("file://")) return false;
    if (!url.endsWith(".pdf")) return false;

    const filePath = decodeURIComponent(url).replace("file://", "");
    const storedPaths = Object.entries(global.state.files).filter(
        ([id, file]) => file.filename === filePath
    );

    if (storedPaths.length > 0) {
        console.log("Found stored");
        return storedPaths[0][0];
    }

    const filename = decodeURIComponent(url.split("/").last())
        .toLowerCase()
        .replace(/\W/g, "");
    const titles = Object.values(cleanPapers(global.state.papers))
        .map((p) => {
            return { title: p.title.toLowerCase().replace(/\W/g, ""), id: p.id };
        })
        .filter((t) => filename.includes(t.title));

    if (titles.length === 0) return false;

    return titles[0].id;
};
