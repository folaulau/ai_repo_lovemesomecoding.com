// Browser smoke test for the RAG test console. Needs the API (8010) and `npm run dev` (5180) running.
//   node e2e/smoke.mjs            search-only flows (free)
//   node e2e/smoke.mjs --ask      also one real Claude question (~$0.02)
// Screenshots land in e2e/screenshots/ (gitignored).
import { mkdirSync } from 'node:fs'
import { chromium } from 'playwright'

const BASE = 'http://localhost:5180'
const SHOTS = new URL('./screenshots/', import.meta.url).pathname
const withAsk = process.argv.includes('--ask')
mkdirSync(SHOTS, { recursive: true })

const failures = []
const check = (ok, msg) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${msg}`)
  if (!ok) failures.push(msg)
}

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1200, height: 900 } })
const consoleErrors = []
page.on('console', (m) => m.type() === 'error' && consoleErrors.push(m.text()))

await page.goto(BASE)
await page.getByText(/11,701 chunks/).waitFor({ timeout: 15000 })
check(true, 'health loaded: chunk count shown')

// 1. Search only: the lesson 3 fix should hold in the UI too.
await page.getByRole('button', { name: 'What is a Python decorator?' }).click()
await page.getByRole('button', { name: 'Search only (free)' }).click()
await page.locator('.source').first().waitFor()
const firstUrl = await page.locator('.source .url').first().innerText()
check(firstUrl === '/python/python-decorators', `top source is the Python post (got ${firstUrl})`)
check((await page.locator('.source').count()) === 5, 'five sources by default')
await page.screenshot({ path: SHOTS + '1-search.png', fullPage: true })

// 2. Top k and model switch.
await page.getByLabel('Top k').selectOption('3')
await page.getByLabel('Embedding').selectOption('base')
await page.getByRole('button', { name: 'Search only (free)' }).click()
await page.waitForFunction(() => document.querySelectorAll('.source').length === 3)
check(true, 'top k = 3 returns three sources with bge-base')
await page.getByLabel('Top k').selectOption('5')
await page.getByLabel('Embedding').selectOption('small')

// 3. Out-of-scope question shows the low-score warning.
await page.getByRole('button', { name: /Kubernetes operator/ }).click()
await page.getByRole('button', { name: 'Search only (free)' }).click()
await page.getByText(/the site may not cover this/).waitFor({ timeout: 10000 })
check(true, 'low-score warning shown for a question the site does not cover')
await page.screenshot({ path: SHOTS + '2-out-of-scope.png', fullPage: true })

// 4. A too-short question explains itself instead of silently doing nothing.
await page.getByLabel('Question').fill('hi')
await page.getByRole('button', { name: 'Search only (free)' }).click()
await page.locator('.error').waitFor()
check(/at least 3 characters/.test(await page.locator('.error').innerText()), 'too-short question shows an error')

// 5. Real answer with clickable citations.
if (withAsk) {
  await page.getByRole('button', { name: 'How do I create a thread in Java?' }).click()
  await page.getByRole('button', { name: /Ask Claude/ }).click()
  await page.locator('.answer').waitFor({ timeout: 120000 })
  const cites = page.locator('.answer a.cite')
  check((await cites.count()) > 0, `answer has citation links (${await cites.count()})`)
  check(/\$0\.\d{4}/.test(await page.locator('.meta').innerText()), 'cost shown in meta line')
  const target = (await cites.first().getAttribute('href')).slice(1)
  await cites.first().click()
  await page.locator(`#${target}.hl`).waitFor({ timeout: 5000 })
  check(true, `clicking a citation highlights ${target}`)
  await page.screenshot({ path: SHOTS + '3-answer.png', fullPage: true })
}

check(consoleErrors.length === 0, `no console errors${consoleErrors.length ? ': ' + consoleErrors.join(' | ') : ''}`)
await browser.close()
console.log(failures.length ? `\n${failures.length} failed` : '\nall passed')
process.exit(failures.length ? 1 : 0)
