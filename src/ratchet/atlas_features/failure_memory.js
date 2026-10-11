// Atlas Trigger function: turn one inserted failure into one sanitized failure-memory document.
//
// Idempotent by construction: the memory document's _id is derived from the failure's own id, and the write
// is an upsert with $setOnInsert, so a duplicate delivery of the same change event (Triggers deliver at least
// once) or a second insert of the same failure writes nothing new. The text is embedded by the collection's
// autoEmbed vector index, so this function holds no embedding key.

const MEMORY_DB = "ratchet_featureprobe";
const MEMORY_COLLECTION = "failure_memory";

function sanitize(text) {
  return String(text)
    .replace(/[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g, "<email>")
    .replace(/\b(?:ch|re|ord|cus|ti|tk|sub|pi|in|acct)_[A-Za-z0-9]+\b/g, "<id>")
    .replace(/\b(order|ticket|charge|invoice|account|customer)\s+#?[A-Za-z]*\d[A-Za-z0-9-]*\b/gi, "$1 <id>")
    .replace(/#\d+\b/g, "<id>")
    .replace(/\$\s?\d[\d,]*(?:\.\d+)?/g, "<amount>")
    .replace(/\b\d[\d,]*(?:\.\d+)?\s*(?:dollars|usd|eur|cents)\b/gi, "<amount>")
    .replace(/\s+/g, " ")
    .trim();
}

function memoryDocument(failure) {
  return {
    _id: "fm:" + failure.failure_id,
    failure_id: failure.failure_id,
    case_family: failure.case_family || null,
    harness_version: failure.harness_version || null,
    summary: sanitize(failure.summary),
    source: "trigger",
  };
}

exports = async function (changeEvent) {
  const failure = changeEvent.fullDocument;
  if (!failure || !failure.failure_id || !failure.summary) {
    return { skipped: "missing failure_id or summary" };
  }
  const memory = context.services.get("mongodb-atlas").db(MEMORY_DB).collection(MEMORY_COLLECTION);
  const doc = memoryDocument(failure);
  const { _id, ...fields } = doc;
  const result = await memory.updateOne(
    { _id },
    { $setOnInsert: { ...fields, created_at: new Date() } },
    { upsert: true }
  );
  return { id: _id, inserted: result.upsertedId !== undefined && result.upsertedId !== null };
};

if (typeof module !== "undefined") {
  module.exports = { sanitize, memoryDocument, handler: exports };
}
