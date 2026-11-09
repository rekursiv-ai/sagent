/**
 * @file generate-rekursiv-skills.mjs
 * Generates bundled Rekursiv AI skills for the VS Code extension at build time.
 * Resolves skills from local sibling repositories if present; falls back to Git sparse-checkout.
 */

import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { copyFile, mkdir, mkdtemp, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { promisify } from "node:util";

const executeProcess = promisify(execFile);

/**
 * Configuration definition for a source repository providing Rekursiv AI skills.
 */
class RekursivSkillBundleDefinition {
  /**
   * @param {Object} options
   * @param {string} options.bundleId - Identifier used for bundle directory naming (e.g., 'trackinizer').
   * @param {string} options.repositoryDisplayName - Human-readable repository title for manifest and logs.
   * @param {string} options.gitRemoteUrl - HTTPS Git repository URL for sparse cloning.
   * @param {string} [options.gitBranch='main'] - Target Git branch.
   * @param {string} options.skillsSubdirectory - Relative path inside repository containing SKILL.md files.
   * @param {string[]} options.localSiblingPathParts - Relative path components from local workspace root.
   */
  constructor({
    bundleId,
    repositoryDisplayName,
    gitRemoteUrl,
    gitBranch = "main",
    skillsSubdirectory,
    localSiblingPathParts,
  }) {
    this.bundleId = bundleId;
    this.repositoryDisplayName = repositoryDisplayName;
    this.gitRemoteUrl = gitRemoteUrl;
    this.gitBranch = gitBranch;
    this.skillsSubdirectory = skillsSubdirectory;
    this.localSiblingPathParts = localSiblingPathParts;
  }
}

/**
 * Resolves and copies skill files from local disk or clones them on demand using Git.
 */
class RekursivSkillDownloader {
  /**
   * @param {string} localWorkspaceRoot - Path to parent workspace directory where sibling repos might reside.
   */
  constructor(localWorkspaceRoot) {
    this.localWorkspaceRoot = localWorkspaceRoot;
  }

  /**
   * Resolves the source directory containing skill files for a given bundle definition.
   * Checks local sibling repository first; falls back to Git sparse-checkout if missing.
   *
   * @param {RekursivSkillBundleDefinition} bundleDefinition
   * @returns {Promise<{ sourceDirectory: string, cleanupCallback: (() => Promise<void>) | null }>}
   */
  async resolveSkillSourceDirectory(bundleDefinition) {
    // 1. Try local sibling repository if available
    const localSkillPath = join(this.localWorkspaceRoot, ...bundleDefinition.localSiblingPathParts);
    if (existsSync(localSkillPath)) {
      console.log(`[skills] Found local skills for "${bundleDefinition.repositoryDisplayName}" at: ${localSkillPath}`);
      return { sourceDirectory: localSkillPath, cleanupCallback: null };
    }

    // 2. Fetch from Git via shallow sparse-checkout into a temporary directory
    console.log(`[skills] Fetching "${bundleDefinition.repositoryDisplayName}" from Git (${bundleDefinition.gitRemoteUrl})...`);
    const temporaryCloneDirectory = await mkdtemp(join(tmpdir(), `rekursiv-skill-${bundleDefinition.bundleId}-`));

    try {
      await executeProcess("git", [
        "clone",
        "--depth", "1",
        "--branch", bundleDefinition.gitBranch,
        "--filter=blob:none",
        "--sparse",
        bundleDefinition.gitRemoteUrl,
        temporaryCloneDirectory,
      ]);

      await executeProcess("git", ["sparse-checkout", "set", bundleDefinition.skillsSubdirectory], {
        cwd: temporaryCloneDirectory,
      });

      const resolvedSkillDirectory = join(temporaryCloneDirectory, bundleDefinition.skillsSubdirectory);
      if (!existsSync(resolvedSkillDirectory)) {
        throw new Error(
          `Subdirectory "${bundleDefinition.skillsSubdirectory}" not found in clone of ${bundleDefinition.gitRemoteUrl}`,
        );
      }

      return {
        sourceDirectory: resolvedSkillDirectory,
        cleanupCallback: async () => {
          await rm(temporaryCloneDirectory, { recursive: true, force: true });
        },
      };
    } catch (error) {
      await rm(temporaryCloneDirectory, { recursive: true, force: true });
      throw new Error(
        `Failed to fetch skills for "${bundleDefinition.repositoryDisplayName}" from Git: ${error.message}`,
      );
    }
  }

  /**
   * Recursively copies all SKILL.md files from source to destination preserving directory hierarchy.
   *
   * @param {string} sourceDirectory - Source directory to search.
   * @param {string} destinationDirectory - Target directory to write to.
   */
  async copySkillMarkdownFilesRecursively(sourceDirectory, destinationDirectory) {
    const directoryEntries = await readdir(sourceDirectory, { withFileTypes: true });

    for (const entry of directoryEntries) {
      const sourceEntryPath = join(sourceDirectory, entry.name);
      const destinationEntryPath = join(destinationDirectory, entry.name);

      if (entry.isDirectory()) {
        await this.copySkillMarkdownFilesRecursively(sourceEntryPath, destinationEntryPath);
      } else if (entry.isFile() && entry.name === "SKILL.md") {
        await mkdir(destinationDirectory, { recursive: true });
        await copyFile(sourceEntryPath, destinationEntryPath);
      }
    }
  }
}

/**
 * Orchestrates bundling Rekursiv skills and writing the bundle manifest.
 */
class RekursivSkillBundleGenerator {
  /**
   * @param {Object} options
   * @param {string} options.projectRoot - Base path of the VS Code extension project.
   * @param {RekursivSkillBundleDefinition[]} options.bundleDefinitions - List of bundles to build.
   */
  constructor({ projectRoot, bundleDefinitions }) {
    this.projectRoot = projectRoot;
    this.bundleDefinitions = bundleDefinitions;
    this.outputDirectory = join(projectRoot, "resources", "rekursiv-skills");

    const localWorkspaceRoot = process.env.REKURSIV_REPOS_ROOT
      ? resolve(process.env.REKURSIV_REPOS_ROOT)
      : resolve(projectRoot, "../../..");

    this.downloader = new RekursivSkillDownloader(localWorkspaceRoot);
  }

  /**
   * Generates all skill bundles and writes manifest.json.
   */
  async generateAllBundles() {
    console.log(`[skills] Generating Rekursiv skill bundles in: ${this.outputDirectory}`);

    const completedBundles = [];
    const temporaryCleanups = [];

    try {
      await rm(this.outputDirectory, { recursive: true, force: true });
      await mkdir(this.outputDirectory, { recursive: true });

      for (const bundleDefinition of this.bundleDefinitions) {
        try {
          const { sourceDirectory, cleanupCallback } =
            await this.downloader.resolveSkillSourceDirectory(bundleDefinition);

          if (cleanupCallback) {
            temporaryCleanups.push(cleanupCallback);
          }

          const targetBundleDirectory = join(this.outputDirectory, bundleDefinition.bundleId);
          await this.downloader.copySkillMarkdownFilesRecursively(
            sourceDirectory,
            targetBundleDirectory,
          );

          completedBundles.push({
            id: bundleDefinition.bundleId,
            repository: bundleDefinition.repositoryDisplayName,
          });
        } catch (bundleError) {
          console.warn(
            `[skills] Warning: Failed to retrieve bundle "${bundleDefinition.bundleId}": ${bundleError.message}`,
          );
        }
      }

      if (completedBundles.length === 0) {
        throw new Error("No Rekursiv skill bundles could be generated. Check network or local repository paths.");
      }

      const manifestContent = {
        source: "native Rekursiv repositories",
        bundles: completedBundles,
      };

      await writeFile(
        join(this.outputDirectory, "manifest.json"),
        `${JSON.stringify(manifestContent, null, 2)}\n`,
        "utf8",
      );

      console.log(`[skills] Successfully generated ${completedBundles.length} skill bundle(s) in ${this.outputDirectory}.`);
    } finally {
      for (const cleanup of temporaryCleanups) {
        try {
          await cleanup();
        } catch (cleanupError) {
          console.warn(`[skills] Warning: Failed to clean temporary directory: ${cleanupError.message}`);
        }
      }
    }
  }
}

// Bundle configurations
const REKURSIV_BUNDLE_DEFINITIONS = [
  new RekursivSkillBundleDefinition({
    bundleId: "trackinizer",
    repositoryDisplayName: "Trackinizer",
    gitRemoteUrl: "https://github.com/rekursiv-ai/trackinizer.git",
    gitBranch: "main",
    skillsSubdirectory: "trackinizer/trax/docs/skills",
    localSiblingPathParts: ["trackinizer", "trackinizer", "trax", "docs", "skills"],
  }),
  new RekursivSkillBundleDefinition({
    bundleId: "priml",
    repositoryDisplayName: "Priml",
    gitRemoteUrl: "https://github.com/rekursiv-ai/priml.git",
    gitBranch: "main",
    skillsSubdirectory: "docs",
    localSiblingPathParts: ["priml", "docs"],
  }),
  new RekursivSkillBundleDefinition({
    bundleId: "configgle",
    repositoryDisplayName: "Configgle",
    gitRemoteUrl: "https://github.com/rekursiv-ai/configgle.git",
    gitBranch: "main",
    skillsSubdirectory: "docs",
    localSiblingPathParts: ["configgle", "docs"],
  }),
];

const extensionProjectRoot = resolve(import.meta.dirname, "..");
const generator = new RekursivSkillBundleGenerator({
  projectRoot: extensionProjectRoot,
  bundleDefinitions: REKURSIV_BUNDLE_DEFINITIONS,
});

await generator.generateAllBundles();
