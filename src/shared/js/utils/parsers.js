// -------------------
// -----  Utils  -----
// -------------------

const extractBibtexValue = (bibtex, key) => {
    const regex = new RegExp(`${key}\\s?=\\s?{(.+)},`, "gi");
    const match = regex.exec(bibtex);
    if (match) {
        const regex2 = new RegExp(`${key}\\s?=\\s?{`, "gi");
        return match[0].replace(regex2, "").slice(0, -2);
    }
    return "";
};

const extractAuthor = (bibtex) =>
    extractBibtexValue(bibtex, "author")
        .replaceAll("{", "")
        .replaceAll("}", "")
        .replaceAll("\\", "")
        .split(" and ")
        .map((a) => a.split(", ").reverse().join(" "))
        .join(" and ");

const decodeHtml = (html) => {
    // https://stackoverflow.com/questions/5796718/html-entity-decode
    var txt = document.createElement("textarea");
    txt.innerHTML = html;
    return txt.value;
};

// -------------------
// -----  Fetch  -----
// -------------------

const fetchArxivXML = async (paperId) => {
    const arxivId = paperId.replace("Arxiv-", "");
    return fetch(
        "https://export.arxiv.org/api/query?" +
            new URLSearchParams({ id_list: arxivId })
    );
};

const fetchNeuripsHTML = async (url) => {
    let paperPage;
    if (url.endsWith(".pdf")) {
        paperPage = url
            .replace("/file/", "/hash/")
            .replace("-Paper.pdf", "-Abstract.html");
    } else {
        paperPage = url;
    }

    return fetch(paperPage).then((response) => {
        return response.text();
    });
};

const fetchCvfHTML = async (url) => {
    let paperPage, text;
    if (url.endsWith(".pdf")) {
        paperPage = url
            .replace("/papers_backup/", "/papers/")
            .replace("/papers/", "/html/")
            .replace(".pdf", ".html");
    } else {
        paperPage = url;
    }

    text = await fetch(paperPage).then((response) => {
        return response.ok ? response.text() : "";
    });

    if (!text && paperPage.includes("thecvf.com/content_")) {
        const { conf, year } = parseCVFUrl(url);
        paperPage = paperPage.replace(
            `/content_${conf}_${year}/`,
            `/content_${conf.toLowerCase()}_${year}/`
        );
        text = await fetch(paperPage).then((response) => {
            return response.ok ? response.text() : "";
        });
    }

    return text;
};

const fetchOpenReviewNoteJSON = async (url) => {
    const id = url.match(/id=([\w-])+/)[0].replace("id=", "");
    const api = `https://api.openreview.net/notes?id=${id}`;
    return fetch(api).then((response) => {
        return response.json();
    });
};
const fetchOpenReviewForumJSON = async (url) => {
    const id = url.match(/id=([\w-])+/)[0].replace("id=", "");
    const api = `https://api.openreview.net/notes?forum=${id}`;
    return fetch(api).then((response) => {
        return response.json();
    });
};

// -------------------
// -----  Parse  -----
// -------------------

const makeArxivPaper = async (memoryId) => {
    const response = await fetchArxivXML(memoryId);
    const xmlData = await response.text();
    console.log("xmlData: ", xmlData);
    var doc = new DOMParser().parseFromString(xmlData.replaceAll("\n", ""), "text/xml");

    const authors = Array.from(doc.querySelectorAll("author name")).map(
        (el) => el.innerHTML
    );
    const author = authors.join(" and ");

    let pdfLink = Array.from(doc.getElementsByTagName("link"))
        .map((l) => l.getAttribute("href"))
        .filter((h) => h.includes("arxiv.org/pdf/"))[0];
    const pdfVersion = pdfLink.match(/v\d+\.pdf/gi);
    if (pdfVersion && pdfVersion.length > 0) {
        pdfLink = pdfLink.replace(pdfVersion[0], ".pdf");
    }

    const title = doc.querySelector("entry title").innerHTML;
    const year = doc.querySelector("entry published").innerHTML.slice(0, 4);
    const key =
        authors[0].split(" ").reverse()[0].toLowerCase() +
        year +
        firstNonStopLowercase(title);

    const id = memoryId;
    const conf = "arXiv";

    let bibtex = "";
    bibtex += `@article{${key},\n`;
    bibtex += `    title={${title} },\n`;
    bibtex += `    author={${author} },\n`;
    bibtex += `    year={${year}},\n`;
    bibtex += `    journal={arXiv preprint arXiv: ${id}}\n`;
    bibtex += `}`;

    return { author, bibtex, conf, id, key, pdfLink, title, year };
};

const makeNeuripsPaper = async (url) => {
    const htmlText = await fetchNeuripsHTML(url);
    const doc = new DOMParser().parseFromString(
        htmlText.replaceAll("\n", ""),
        "text/html"
    );

    const paragraphs = Array.from(doc.querySelectorAll(".container-fluid .col p"));
    const hash = url.split("/").slice(-1)[0].replace("-Paper.pdf", "");

    const title = doc.getElementsByTagName("h4")[0].innerHTML;
    const author = paragraphs[1]
        .getElementsByTagName("i")[0]
        .innerHTML.split(", ")
        .map((author, k) => {
            const parts = author.split(" ");
            const caps = parts.map((part, i) => {
                return capitalize(part);
            });
            return caps.join(" ");
        })
        .join(" and ");
    const pdfLink = url;
    const year = paragraphs[0].innerHTML.match(/\d{4}/)[0];
    const key = `neurips${year}${hash.slice(0, 8)}`;
    const id = `NeurIPS-${year}_${hash.slice(0, 8)}`;
    const conf = `NeurIPS ${year}`;
    const note = `Accepted @ ${conf}`;

    let bibtex = "";

    bibtex += `@inproceedings{NEURIPS${year}_${hash.slice(0, 8)},\n`;
    bibtex += `    author={${author}},\n`;
    bibtex += `    booktitle={Advances in Neural Information Processing Systems},\n`;
    bibtex += `    editor={H.Larochelle and M.Ranzato and R.Hadsell and M.F.Balcan and H.Lin},\n`;
    bibtex += `    publisher={Curran Associates, Inc.},\n`;
    bibtex += `    title={${title}},\n`;
    bibtex += `    url={${url}},\n`;
    bibtex += `    year={${year}}\n`;
    bibtex += `}`;
    bibtex = bibtexToString(bibtex);

    return { author, bibtex, conf, id, key, note, pdfLink, title, year };
};

const makeCVFPaper = async (url) => {
    const htmlText = await fetchCvfHTML(url);
    const doc = new DOMParser().parseFromString(
        htmlText.replaceAll("\n", ""),
        "text/html"
    );
    const title = doc.getElementById("papertitle").innerText.trim();
    let author = doc
        .querySelector("#authors i")
        .innerText.split(",")
        .map((a) => a.trim())
        .join(" and ");
    const { year, id, conf } = parseCVFUrl(url);
    let pdfLink = "";
    if (url.endsWith(".pdf")) {
        pdfLink = url;
    } else {
        const href = Array.from(doc.getElementsByTagName("a"))
            .filter((a) => a.innerText === "pdf")[0]
            .getAttribute("href");
        if (href.startsWith("../")) {
            href = href.replaceAll("../", "");
        }
        if (!href.startsWith("/")) {
            href = "/" + href;
        }
        pdfLink = "http://openaccess.thecvf.com" + href;
    }
    const note = `Accepted @ ${conf} ${year}`;
    const bibtex = bibtexToString(doc.querySelector(".bibref").innerText);
    const key = bibtex.split("{")[1].split(",")[0];

    return { author, bibtex, conf, id, key, note, pdfLink, title, year };
};

const makeOpenReviewBibTex = (paper, url) => {
    const title = paper.content.title;
    const author = paper.content.authors.join(" and ");
    const year = paper.cdate ? new Date(paper.cdate).getFullYear() : "0000";
    if (!paper.cdate) {
        log("makeOpenReviewBibTex: no cdate found in", paper);
    }

    let key = paper.content.authors[0].split(" ").reverse()[0];
    key += year;
    key += firstNonStopLowercase(title);

    let bibtex = "";
    bibtex += `@inproceedings{${key},\n`;
    bibtex += `    title={${title}},\n`;
    bibtex += `    author={${author}},\n`;
    bibtex += `    year={${year}},\n`;
    bibtex += `    url={${url}},\n`;
    bibtex += `}`;

    return bibtex;
};

const makeOpenReviewPaper = async (url) => {
    const noteJson = await fetchOpenReviewNoteJSON(url);
    const forumJson = await fetchOpenReviewForumJSON(url);

    var paper = noteJson.notes[0];
    log("paper: ", paper);
    var forum = forumJson.notes;
    log("forum: ", forum);

    const title = paper.content.title;
    const author = paper.content.authors.join(" and ");
    const bibtex = bibtexToString(
        paper.content._bibtex || makeOpenReviewBibTex(paper, url)
    );
    const key = bibtex.split("{")[1].split(",")[0].trim();
    const year = bibtex.split("year={")[1].split("}")[0];

    let pdfLink;
    if (paper.pdf) {
        pdfLink = `https://openreview.net/pdf?id=${paper.id}`;
    } else {
        if (paper.html) {
            pdfLink = paper.html.replace("/forum?id=", "/pdf?id=");
        } else {
            pdfLink = url.replace("/forum?id=", "/pdf?id=");
        }
    }

    const confParts = paper.invitation.split("/");
    let organizer = confParts[0].split(".")[0];
    let event = confParts
        .slice(1)
        .join("/")
        .split("-")[0]
        .replaceAll("/", " ")
        .replace(" Conference", "");

    let overrideOrg = organizer;
    let overridden = false;
    if (global.overrideORConfs.hasOwnProperty(organizer)) {
        overrideOrg = global.overrideORConfs[organizer];
        overridden = true;
    }
    if (overridden) {
        event = event.replace(overrideOrg, "");
        organizer = overrideOrg;
    }

    const conf = `${organizer} ${event}`
        .replace(/ \d\d\d\d/g, "")
        .replace(/\s\s+/g, " ");
    const id = `OR-${organizer}-${year}_${paper.id}`;

    let candidates, decision, note;

    candidates = forum.filter((r) => {
        return (
            r &&
            r.content &&
            ["Final Decision", "Paper Decision", "Acceptance Decision"].indexOf(
                r.content.title
            ) > -1
        );
    });
    log("forum: ", forum);
    if (candidates && candidates.length > 0) {
        decision = candidates[0].content.decision
            .split(" ")
            .map((v, i) => {
                return i === 0 ? v + "ed" : v;
            })
            .join(" ");
        note = `${decision} @ ${conf} ${year}`;
    }

    if (author === "Anonymous") {
        note = `Under review @ ${conf} ${year} (${new Date().toLocaleDateString()})`;
    }

    return { author, bibtex, conf, id, key, note, pdfLink, title, year };
};

const makeBioRxivPaper = async (url) => {
    const biorxivAPI = "https://api.biorxiv.org/";
    const pageURL = url.replace(".full.pdf", "");
    const biorxivID = url
        .split("/")
        .slice(-2)
        .join("/")
        .replace(".full.pdf", "")
        .split("v")[0];
    const api = `${biorxivAPI}/details/biorxiv/${biorxivID}`;
    const data = await fetch(api).then((response) => {
        return response.json();
    });

    if (data.messages[0].status !== "ok")
        throw new Error(`${api} returned ${data.messages[0].status}`);

    const paper = data.collection.reverse()[0];

    const pageData = await fetch(pageURL);
    const pageText = await pageData.text();
    const dom = new DOMParser().parseFromString(
        pageText.replaceAll("\n", ""),
        "text/html"
    );
    const bibtextLink = dom.querySelector(".bibtext a").href;
    const bibtex = bibtexToString(await (await fetch(bibtextLink)).text());

    const author = extractAuthor(bibtex);

    const conf = "BioRxiv";
    const id = parseIdFromUrl(url);
    const key = bibtex.split("\n")[0].split("{")[1].replace(",", "").trim();
    const note = "";
    const pdfLink = cleanBiorxivURL(url) + ".full.pdf";
    const title = paper.title;
    const year = paper.date.split("-")[0];

    return { author, bibtex, conf, id, key, note, pdfLink, title, year };
};

const makePMLRPaper = async (url) => {
    const key = url.split("/").reverse()[0].split(".")[0];
    const id = parseIdFromUrl(url);

    const absURL = url.includes(".html")
        ? url
        : url.split("/").slice(0, -2).join("/") + `${key}.html`;

    const pdfLink = absURL.replace(".html", "") + `/${key}.pdf`;

    const doc = new DOMParser().parseFromString(
        (await (await fetch(absURL)).text()).replaceAll("\n", ""),
        "text/html"
    );

    const bibURL = doc
        .getElementById("button-bibtex1")
        .getAttribute("onclick")
        .match(/https.+\.bib/)[0];
    const bibtexRaw = doc
        .getElementById("bibtex")
        .innerText.replaceAll("\t", " ")
        .replaceAll(/\s\s+/g, " ");
    let bibtex = bibtexRaw;
    const items = bibtexRaw.match(/,\ ?\w+ ?= ?{/g);
    for (const item of items) {
        bibtex = bibtex.replace(
            item,
            item.replace(", ", ",\n    ").replace(" = ", "=")
        );
    }
    if (bibtex.endsWith("}}")) {
        bibtex = bibtex.slice(0, -2) + "}\n}";
    }
    bibtex = bibtexToString(bibtex);

    const author = extractAuthor(bibtex);
    const title = doc.getElementsByTagName("h1")[0].innerText;
    const year = extractBibtexValue(bibtex, "year");

    let conf = extractBibtexValue(bibtex, "booktitle").replaceAll(
        "Proceedings of the",
        ""
    );
    note = "Accepted @ " + conf + ` (${year})`;
    for (const long in global.overridePMLRConfs) {
        if (conf.includes(long)) {
            conf = global.overridePMLRConfs[long] + " " + year;
            note = "Accepted @ " + conf;
            break;
        }
    }

    return { author, bibtex, conf, id, key, note, pdfLink, title, year };
};

const findACLValue = (dom, key) => {
    const dt = Array.from(dom.querySelectorAll("dt")).filter((v) =>
        v.innerText.includes(key)
    )[0];
    return dt.nextElementSibling.innerText;
};

const makeACLPaper = async (url) => {
    url = url.replace(".pdf", "");
    const htmlText = await fetch(url).then((r) => r.text());
    const dom = new DOMParser().parseFromString(
        htmlText.replaceAll("\n", ""),
        "text/html"
    );

    const bibtexEl = dom.getElementById("citeBibtexContent");
    if (!bibtexEl) return;

    const title = dom.getElementById("title").innerText;
    const bibtex = bibtexToString(bibtexEl.innerText);

    const bibtexData = bibtexToJson(bibtex)[0];
    const entries = bibtexData.entryTags;

    const year = entries.year;
    const author = entries.author
        .replace(/\s+/g, " ")
        .split(" and ")
        .map((v) =>
            v
                .split(",")
                .map((a) => a.trim())
                .reverse()
                .join(" ")
        )
        .join(" and ");
    const key = bibtexData.citationKey;

    const conf = findACLValue(dom, "Venue");
    const pdfLink = findACLValue(dom, "PDF");
    const aid = findACLValue(dom, "Anthology ID");

    const id = `ACL-${conf}-${year}_${aid}`;
    const note = `Accepted @ ${conf} ${year}`;

    return { author, bibtex, conf, id, key, note, pdfLink, title, year };
};

const makePNASPaper = async (url) => {
    url = url.replace(".full.pdf", "");
    const htmlText = await fetch(url).then((r) => r.text());
    const dom = new DOMParser().parseFromString(
        htmlText.replaceAll("\n", ""),
        "text/html"
    );
    const citeUrl = dom
        .getElementsByClassName("pane-jnl-pnas-cite-tool")[0]
        .querySelector("a").href;

    if (!citeUrl) return;

    const title = dom.getElementById("page-title").innerText;
    const bibtexUrl = citeUrl.replace("/download", "/bibtext");
    const bibtex = await fetch(bibtexUrl).then((r) => r.text());
    const bibtexData = bibtexToJson(bibtex)[0];

    const entries = bibtexData.entryTags;

    const year = entries.year;
    const author = entries.author
        .replace(/\s+/g, " ")
        .split(" and ")
        .map((v) =>
            v
                .split(",")
                .map((a) => a.trim())
                .reverse()
                .join(" ")
        )
        .join(" and ");
    const pdfLink = entries.eprint;
    const key = bibtexData.citationKey;
    const note = `Published @ PNAS (${year})`;
    const pid = url.endsWith("/")
        ? url.split("/").slice(-2)[0]
        : url.split("/").slice(-1)[0];

    const id = `PNAS-${year}_${pid}`;

    return { author, bibtex, id, key, note, pdfLink, title, year };
};

// --------------------------------------------
// -----  Try CrossRef's API for a match  -----
// --------------------------------------------
/**
 * Looks for a title in crossref's database, querying titles and looking for an exact match. If no
 * exact match is found, it will return an empty note "". If a match is found and `item.event.name`
 * exists, it will be used for a new note.
 * @param {object} paper The paper to look for in crossref's database for an exact ttile match
 * @returns {string} The note for the paper as `Accepted @ ${items.event.name} -- [crossref.org]`
 */
const tryCrossRef = async (paper) => {
    try {
        // fetch crossref' api for the paper's title
        const title = encodeURI(paper.title);
        const api = `https://api.crossref.org/works?rows=1&mailto=schmidtv%40mila.quebec&select=event%2Ctitle&query.title=${title}`;
        const json = await fetch(api).then((response) => response.json());

        // assert the response is valid
        if (json.status !== "ok") {
            log(`[PM][Crossref] ${api} returned ${json.message.status}`);
            return "";
        }
        // assert there is a (loose) match
        if (json.message.items.length === 0) return "";

        // compare matched item's title to the paper's title
        const crossTitle = json.message.items[0].title[0]
            .toLowerCase()
            .replaceAll("\n", " ")
            .replaceAll(/\s\s+/g, " ");
        const refTitle = paper.title
            .toLowerCase()
            .replaceAll("\n", " ")
            .replaceAll(/\s\s+/g, " ");
        if (crossTitle !== refTitle) {
            return "";
        }

        // assert the matched item has an event with a name
        // (this may be too restrictive for journals, to improve)
        if (!json.message.items[0].event || !json.message.items[0].event.name)
            return "";

        // return the note
        info("Found a CrossRef match");
        return `Accepted @ ${json.message.items[0].event.name.trim()} -- [crossref.org]`;
    } catch (error) {
        // something went wrong, log the error, return ""
        log("[PM][Crossref]", error);
        return "";
    }
};

const tryDBLP = async (paper) => {
    try {
        const title = encodeURI(paper.title);
        const api = `https://dblp.org/search/publ/api?q=${title}&format=json`;
        const json = await fetch(api).then((response) => response.json());

        if (
            !json.result ||
            !json.result.hits ||
            !json.result.hits.hit ||
            !json.result.hits.hit.length
        ) {
            log("[PM][DBLP] No hits found");
            return "";
        }

        const hits = json.result.hits.hit.sort(
            (a, b) => parseInt(a.info.year, 10) - parseInt(b.info.year, 10)
        );

        for (const hit of hits) {
            const hitTitle = decodeHtml(
                hit.info.title
                    .toLowerCase()
                    .replaceAll("\n", " ")
                    .replaceAll(".", "")
                    .replaceAll(/\s\s+/g, " ")
            );
            const refTitle = paper.title
                .toLowerCase()
                .replaceAll("\n", " ")
                .replaceAll(".", "")
                .replaceAll(/\s\s+/g, " ");
            if (hitTitle === refTitle && hit.info.venue !== "CoRR") {
                info("Found a DBLP match");
                const abbr = hit.info.venue.toLowerCase().replaceAll(".", "").trim();
                const venue = global.journalAbbreviations[abbr] || hit.info.venue;
                const year = hit.info.year;
                const url = hit.info.url;
                const note = `Accepted @ ${venue.trim()} ${year} -- [dblp.org]\n${url}`;
                return note;
            }
        }
        log("[PM][DBLP] No match found");
        return "";
    } catch (error) {
        // something went wrong, log the error, return ""
        log("[PM][DBLP]", error);
        return "";
    }
};

const tryPreprintMatch = async (paper) => {
    let note = "";
    note = await tryDBLP(paper);
    if (!note) {
        note = await tryCrossRef(paper);
    }
    return note;
};

// -----------------------------
// -----  Creating papers  -----
// -----------------------------

const initPaper = async (paper) => {
    if (!paper.note) {
        paper.note = "";
    }

    paper.md = `[${paper.title}](${paper.pdfLink})`;
    paper.tags = [];
    paper.codeLink = "";
    paper.favorite = false;
    paper.favoriteDate = "";
    paper.addDate = new Date().toJSON();
    paper.lastOpenDate = paper.addDate;
    paper.count = 1;

    for (const k in paper) {
        if (paper.hasOwnProperty(k) && typeof paper[k] === "string") {
            paper[k] = paper[k].trim();
        }
    }

    paper = await autoTagPaper(paper);
    validatePaper(paper);

    return paper;
};

const autoTagPaper = async (paper) => {
    try {
        const autoTags = await getStorage("autoTags");
        if (!autoTags || !autoTags.length) return paper;
        let tags = new Set();
        for (const at of autoTags) {
            if (!at.tags?.length) continue;
            if (!at.title && !at.author) continue;

            const titleMatch = at.title
                ? new RegExp(at.title, "i").test(paper.title)
                : true;
            const authorMatch = at.author
                ? new RegExp(at.author, "i").test(paper.author)
                : true;

            if (titleMatch && authorMatch) {
                at.tags.forEach((t) => tags.add(t));
            }
        }
        paper.tags = Array.from(tags).sort();
        if (paper.tags.length) {
            log("Automatically adding tags:", paper.tags);
        }
        return paper;
    } catch (error) {
        log("Error auto-tagging:", error);
        log("Paper:", paper);
        return paper;
    }
};

const makePaper = async (is, url, id) => {
    let paper;
    if (is.arxiv) {
        paper = await makeArxivPaper(id);
        paper.source = "arxiv";
        // paper.codes = await fetchCodes(paper)
    } else if (is.neurips) {
        paper = await makeNeuripsPaper(url);
        paper.source = "neurips";
        // paper.codes = await fetchCodes(paper);
    } else if (is.cvf) {
        paper = await makeCVFPaper(url);
        paper.source = "cvf";
    } else if (is.openreview) {
        paper = await makeOpenReviewPaper(url);
        paper.source = "openreview";
    } else if (is.biorxiv) {
        paper = await makeBioRxivPaper(url);
        paper.source = "biorxiv";
    } else if (is.pmlr) {
        paper = await makePMLRPaper(url);
        paper.source = "pmlr";
    } else if (is.acl) {
        paper = await makeACLPaper(url);
        if (paper) {
            paper.source = "pmlr";
        }
    } else if (is.pnas) {
        paper = await makePNASPaper(url);
        if (paper) {
            paper.source = "pnas";
        }
    } else {
        throw new Error("Unknown paper source: " + JSON.stringify({ is, url, id }));
    }

    if (typeof paper === "undefined") {
        return;
    }

    return await initPaper(paper);
};

const findFuzzyPaperMatch = (paper) => {
    for (const paperId in global.state.papers) {
        if (paperId === "__dataVersion") continue;
        const item = global.state.papers[paperId];
        if (
            Math.abs(item.title.length - paper.title.length) <
            global.fuzzyTitleMatchMinDist
        ) {
            const dist = levenshtein(item.title, paper.title);
            if (dist < global.fuzzyTitleMatchMinDist) {
                return item.id;
            }
        }
    }
    return null;
};
