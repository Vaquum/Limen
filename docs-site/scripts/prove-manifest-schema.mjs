import {createRequire} from 'node:module';
import {createHash} from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

const require = createRequire(import.meta.url);
const engine = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../node_modules/yaml-language-server');
const {getLanguageService} = require(engine + '/lib/umd/languageservice/yamlLanguageService.js');
const {parse} = require(engine + '/lib/umd/languageservice/parser/yamlParser07.js');
const {getNodeValue} = require(engine + '/lib/umd/languageservice/utils/astNodeUtils.js');
const {SettingsState} = require(engine + '/lib/umd/yamlSettings.js');
const {TextDocument} = require(path.resolve(engine, '../vscode-languageserver-textdocument'));
const [schemaPath, casesPath, outputPath] = process.argv.slice(2);
if (!schemaPath || !casesPath || !outputPath) throw new Error('Expected installed schema, YAML cases and output paths');
const schemaBytes = await fs.readFile(schemaPath);
const schemaURI = pathToFileURL(path.resolve(schemaPath)).href;
const requests = [];
const settings = new SettingsState();
settings.schemaStoreEnabled = false;
settings.kubernetesCRDStoreEnabled = false;
const service = getLanguageService({
  schemaRequestService: async uri => {
    requests.push(uri);
    if (uri !== schemaURI) throw new Error('Unexpected schema fetch: ' + uri);
    return schemaBytes.toString('utf8');
  },
  workspaceContext: {resolveRelativePath: (ref, resource) => new URL(ref, resource).href},
  telemetry: {sendError: (context, error) => {throw new Error(context, {cause: error});}},
  yamlSettings: settings,
});
service.configure({validate: true, yamlVersion: '1.2', customTags: [], disableAdditionalProperties: false,
  schemas: [{uri: schemaURI, fileMatch: ['*.yaml']}]});

function nonJsonValues(value, locus = '') {
  if (typeof value === 'number' && !Number.isFinite(value)) return [{path: locus, value: String(value)}];
  if (value === undefined) return [{path: locus, value: 'undefined'}];
  if (Array.isArray(value)) return value.flatMap((item, index) => nonJsonValues(item, `${locus}[${index}]`));
  if (value && typeof value === 'object') return Object.entries(value).flatMap(([key, item]) => nonJsonValues(item, locus ? `${locus}.${key}` : key));
  return [];
}
const cases = [];
for (const {id, yaml} of JSON.parse(await fs.readFile(casesPath, 'utf8'))) {
  const document = TextDocument.create(`file:///limen-schema-proof/${id}.yaml`, 'yaml', 1, yaml);
  const parsed = parse(yaml, {customTags: [], yamlVersion: '1.2'}, document);
  const models = parsed.documents.map(item => getNodeValue(item.root));
  const nonJson = nonJsonValues(models);
  const parserDiagnostics = parsed.documents.flatMap(item => [...item.errors, ...item.warnings]);
  const diagnostics = await service.doValidation(document, false);
  cases.push({id, yaml_sha256: createHash('sha256').update(yaml).digest('hex'),
    json_compatible: nonJson.length === 0, non_json_values: nonJson,
    ...(nonJson.length === 0 ? {models} : {}), parser_diagnostics: parserDiagnostics, diagnostics});
}
const version = JSON.parse(await fs.readFile(engine + '/package.json', 'utf8')).version;
await fs.writeFile(outputPath, JSON.stringify({engine: {name: 'yaml-language-server', version, node: process.version},
  dialect: JSON.parse(schemaBytes)['$schema'], yaml_version: '1.2', schema_uri: schemaURI,
  schema_sha256: createHash('sha256').update(schemaBytes).digest('hex'), schema_store: false,
  requests, cases}, null, 2) + '\n');
