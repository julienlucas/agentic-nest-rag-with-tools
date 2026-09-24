import { Module } from "@nestjs/common";
import { ServeStaticModule } from "@nestjs/serve-static";
import * as fs from "node:fs";
import { env } from "./config/settings";
import { RagController } from "./api/rag.controller";
import { RagService } from "./api/rag.service";
import { SessionStore } from "./api/session.store";

// En production, Nest sert aussi le frontend buildé (comme Django servait frontend/dist).
const staticModules = fs.existsSync(env.FRONTEND_DIST)
  ? [
      ServeStaticModule.forRoot(
        { rootPath: env.FRONTEND_DIST, exclude: ["/api/{*path}", "/static/{*path}"] },
        { rootPath: env.EXAMPLES_DIR, serveRoot: "/static" },
      ),
    ]
  : [ServeStaticModule.forRoot({ rootPath: env.EXAMPLES_DIR, serveRoot: "/static" })];

@Module({
  imports: [...staticModules],
  controllers: [RagController],
  providers: [RagService, SessionStore],
})
export class AppModule {}
