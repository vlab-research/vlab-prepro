/**
 * Generates test cases for the Python compute_seed function.
 *
 * This uses the canonical JavaScript implementation (farmhash) to generate
 * test fixtures that verify the Python implementation matches exactly.
 *
 * Usage:
 *   cd test_fixtures
 *   npm install
 *   npm run generate
 */
const farmhash = require('farmhash')
const fs = require('fs')
const path = require('path')

function hash(s) {
  return farmhash.fingerprint32(s + '')
}

function getSeed(seed, n, m = 0) {
  for (let i = 0; i < m; i++) {
    seed = hash(seed)
  }
  return (seed % n) + 1
}

// Generate realistic base seeds (as they would come from form+userId)
const formUserCombos = [
  "survey_abc123" + "user_001",
  "survey_abc123" + "user_002",
  "baseline_2024" + "fb_12345678",
  "endline_2024" + "fb_12345678",
  "form1" + "100000000000001",
  "form1" + "100000000000002",
]

const baseSeeds = formUserCombos.map(combo => hash(combo))

// Test different n and m values
const nValues = [2, 3, 5, 7, 10]
const mValues = [0, 1, 2, 3]

const testData = {
  hashTestCases: baseSeeds.map(seed => ({ input: seed, output: hash(seed) })),
  seedTestCases: []
}

for (const seed of baseSeeds) {
  for (const n of nValues) {
    for (const m of mValues) {
      testData.seedTestCases.push({
        seed,
        n,
        m,
        result: getSeed(seed, n, m)
      })
    }
  }
}

const outputPath = path.join(__dirname, '..', 'tests', 'seed_test_cases.json')
fs.writeFileSync(outputPath, JSON.stringify(testData, null, 2))
console.log(`Generated ${testData.seedTestCases.length} test cases to ${outputPath}`)
