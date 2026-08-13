import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Every test mounts into the same document, so a component left behind by one
// test is a component the next one can find. Unmounting between tests keeps a
// passing test from depending on the one before it.
afterEach(cleanup);
