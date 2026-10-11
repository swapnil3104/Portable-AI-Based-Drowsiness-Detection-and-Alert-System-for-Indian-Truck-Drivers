import { test, expect } from '@playwright/test';

test.describe('DrishtiDrive AI Dashboard Smoke Test', () => {
  test('renders full dashboard matching reference design', async ({ page }) => {
    // Navigate to frontend
    await page.goto('http://localhost:3000/');

    // Assert header elements
    await expect(page.locator('.brand-title')).toHaveText('DrishtiDrive AI');
    await expect(page.locator('.brand-tagline')).toHaveText('A Watchful Eye on Every Journey.');
    await expect(page.locator('.status-label')).toHaveText('System Online');

    // Assert Live Driver Monitoring card
    await expect(page.locator('.card-title', { hasText: 'Live Driver Monitoring' })).toBeVisible();
    const startBtn = page.locator('#start-camera-btn');
    if (await startBtn.isVisible()) {
      await startBtn.click();
    }

    // Assert HUD strip items
    await expect(page.locator('.telemetry-hud-strip')).toBeVisible();
    await expect(page.locator('.hud-item', { hasText: 'EAR:' })).toBeVisible();
    await expect(page.locator('.hud-item', { hasText: 'MAR:' })).toBeVisible();

    // Assert Camera Controls
    await expect(page.locator('.card-title', { hasText: 'Camera Controls' })).toBeVisible();
    await expect(page.locator('#camera-select')).toBeVisible();

    // Assert Alert System
    await expect(page.locator('.card-title', { hasText: 'Alert System' })).toBeVisible();
    await expect(page.locator('.stage-card.stage-1')).toBeVisible();
    await expect(page.locator('.stage-card.stage-2')).toBeVisible();
    await expect(page.locator('.stage-card.stage-3')).toBeVisible();

    // Click Preview Alert button
    await page.locator('#preview-alert-btn').click();

    // Assert Driver Status & Gauge
    await expect(page.locator('.card-title', { hasText: 'Driver Status' })).toBeVisible();
    await expect(page.locator('.gauge-container')).toBeVisible();

    // Assert Driver Metrics
    await expect(page.locator('.card-title', { hasText: 'Driver Metrics' })).toBeVisible();
    await expect(page.locator('.metric-label', { hasText: 'Attention Score' })).toBeVisible();
    await expect(page.locator('.metric-label', { hasText: 'Eye Openness (EAR)' })).toBeVisible();
    await expect(page.locator('.metric-label', { hasText: 'Mouth Openness (MAR)' })).toBeVisible();
    await expect(page.locator('.metric-label', { hasText: 'Blink Rate' })).toBeVisible();

    // Assert Event Log
    await expect(page.locator('.card-title', { hasText: 'Event Log' })).toBeVisible();
    await expect(page.locator('#log-entries-scroll')).toBeVisible();

    // Take full page screenshot
    await page.screenshot({ path: '.agent/screenshots/TASK-33-dashboard.png', fullPage: true });
  });
});
